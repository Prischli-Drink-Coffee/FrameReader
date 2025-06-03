import json
import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from ultralytics import YOLO
import triton_python_backend_utils as pb_utils


class YOLOConfig:
    def __init__(
        self, 
        batch_size: int = 16, 
        height: int = 640, 
        width: int = 640,
        conf: float = 0.2,
        iou: float = 0.2,
        rect: bool = True,
        max_det: int = 300,
        classes: Optional[List[int]] = None
    ):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.conf = conf
        self.iou = iou
        self.rect = rect
        self.max_det = max_det
        self.classes = classes


class YOLOProcessor:
    @staticmethod
    def serialize_results(results) -> List[Dict[str, Any]]:
        output = []
        for result in results:
            data = {"boxes": [], "confidences": [], "classes": []}
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if boxes.xyxy is not None and len(boxes.xyxy) > 0:
                    data['boxes'] = boxes.xyxy.cpu().numpy().tolist()
                if boxes.conf is not None and len(boxes.conf) > 0:
                    data['confidences'] = boxes.conf.cpu().numpy().tolist()
                if boxes.cls is not None and len(boxes.cls) > 0:
                    data['classes'] = boxes.cls.cpu().numpy().astype(int).tolist()
            
            output.append(data)
        return output

    @staticmethod
    def preprocess_images(image_data: np.ndarray) -> List[np.ndarray]:
        processed = []
        for i in range(image_data.shape[0]):
            processed.append(image_data[i])
        return processed


class TritonPythonModel:
    def __init__(self):
        self._config: Optional[YOLOConfig] = None
        self._model: Optional[YOLO] = None
        self._device: Optional[torch.device] = None
        self._processor = YOLOProcessor()
        self._params: Dict[str, str] = {}

    def _get_param(self, key: str, default: str = "") -> str:
        return self._params.get(key, default)

    def _get_int_param(self, key: str, default: int) -> int:
        return int(self._get_param(key, str(default)))

    def _get_float_param(self, key: str, default: float) -> float:
        return float(self._get_param(key, str(default)))

    def _get_bool_param(self, key: str, default: bool) -> bool:
        value = self._get_param(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def _get_list_param(self, key: str) -> Optional[List[int]]:
        value = self._get_param(key)
        if not value or value.strip() == "":
            return None
        try:
            return [int(x.strip()) for x in value.split(',') if x.strip()]
        except ValueError:
            return None

    def _parse_parameters(self, config: Dict[str, Any]) -> None:
        self._params = {}
        for param in config.get("parameters", []):
            key = param["key"]
            value = param["value"]["string_value"]
            self._params[key] = value

    def _create_config(self, config_dict: Dict[str, Any]) -> YOLOConfig:
        self._parse_parameters(config_dict)
        
        batch_size = max(int(config_dict.get("max_batch_size", 1)), 1)
        height = self._get_int_param("image_height", 640)
        width = self._get_int_param("image_width", 640)
        conf = self._get_float_param("conf", 0.2)
        iou = self._get_float_param("iou", 0.2)
        rect = self._get_bool_param("rect", True)
        max_det = self._get_int_param("max_det", 300)
        classes = self._get_list_param("classes")
        
        return YOLOConfig(
            batch_size=batch_size,
            height=height,
            width=width,
            conf=conf,
            iou=iou,
            rect=rect,
            max_det=max_det,
            classes=classes
        )

    def initialize(self, args: Dict[str, Any]) -> None:
        try:
            config_dict = json.loads(args["model_config"])
            self._config = self._create_config(config_dict)
            device_id = int(args["model_instance_device_id"])
            
            model_path = os.path.join(
                args["model_repository"], 
                args["model_version"], 
                "yolo_int8.engine"
            )
    
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
                
            self._model = YOLO(model_path)
            self._device = (
                torch.device(f"cuda:{device_id}") 
                if torch.cuda.is_available() 
                else torch.device("cpu")
            )
            
            self._warmup()
            pb_utils.Logger.log_info("YOLO model initialized successfully")
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Initialization failed: {str(e)}")
            raise

    def _warmup(self) -> None:
        dummy = np.zeros((self._config.height, self._config.width, 3), dtype=np.uint8)
        self._model.predict(
            dummy, 
            verbose=False,
            conf=self._config.conf,
            iou=self._config.iou,
            rect=self._config.rect,
            max_det=self._config.max_det,
            classes=self._config.classes
        )

    def execute(self, requests) -> List:
        responses = []
        
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image_data = image_tensor.as_numpy()
                
                pb_utils.Logger.log_info(f"Input shape: {image_data.shape}")
                
                if image_data.size == 0:
                    responses.append(self._create_error_response("Empty input data"))
                    continue
                
                processed_images = self._processor.preprocess_images(image_data)
                
                results = self._model.predict(
                    source=processed_images,
                    conf=self._config.conf,
                    iou=self._config.iou,
                    imgsz=(self._config.height, self._config.width),
                    device=self._device,
                    verbose=False,
                    augment=False,
                    rect=self._config.rect,
                    max_det=self._config.max_det,
                    classes=self._config.classes
                )

                serialized = self._processor.serialize_results(results)
                batch_jsons = [json.dumps(res, ensure_ascii=False, separators=(',', ':')) for res in serialized]
                arr = np.array(batch_jsons, dtype=np.object_)
                output_tensor = pb_utils.Tensor("result", arr)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                
                responses.append(response)
            
            except Exception as e:
                pb_utils.Logger.log_error(f"Request processing failed: {str(e)}")
                responses.append(self._create_error_response(str(e)))
                
        return responses

    def _create_error_response(self, error_message: str) -> pb_utils.InferenceResponse:
        return pb_utils.InferenceResponse(
            error=pb_utils.TritonError(error_message)
        )

    def finalize(self) -> None:
        if self._model is not None:
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pb_utils.Logger.log_info("Model finalized successfully")