import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO
import triton_python_backend_utils as pb_utils
from dataclasses import dataclass


@dataclass
class YOLOConfig:
    batch_size: int = 16
    image_height: int = 640
    image_width: int = 640
    conf: float = 0.2
    iou: float = 0.2
    rect: bool = True
    max_det: int = 300
    classes: Optional[List[int]] = None


class YOLOProcessor:
    @staticmethod
    def serialize_results(results: List) -> List[Dict[str, Any]]:
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
        return [image_data[i] for i in range(image_data.shape[0])]


class ParameterParser:
    def __init__(self, params: Dict[str, str]):
        self._params = params

    def get_param(self, key: str, default: str = "") -> str:
        return self._params.get(key, default)

    def get_int_param(self, key: str, default: int) -> int:
        return int(self.get_param(key, str(default)))

    def get_float_param(self, key: str, default: float) -> float:
        return float(self.get_param(key, str(default)))

    def get_bool_param(self, key: str, default: bool) -> bool:
        value = self.get_param(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def get_list_param(self, key: str) -> Optional[List[int]]:
        value = self.get_param(key)
        if not value or value.strip() == "":
            return None
        try:
            return [int(x.strip()) for x in value.split(',') if x.strip()]
        except ValueError:
            return None


class ConfigParser:
    @staticmethod
    def parse_parameters(config: Dict[str, Any]) -> Dict[str, str]:
        params = {}
        parameters = config.get("parameters", [])
        
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                if isinstance(value, dict) and "string_value" in value:
                    params[key] = value["string_value"]
                else:
                    params[key] = str(value)
        elif isinstance(parameters, list):
            for param in parameters:
                if isinstance(param, dict):
                    key = param.get("key", "")
                    value_dict = param.get("value", {})
                    if isinstance(value_dict, dict) and "string_value" in value_dict:
                        params[key] = value_dict["string_value"]
                    else:
                        params[key] = str(value_dict)
        
        return params

    @staticmethod
    def create_config(config_dict: Dict[str, Any]) -> YOLOConfig:
        params = ConfigParser.parse_parameters(config_dict)
        parser = ParameterParser(params)
        
        batch_size = max(int(config_dict.get("max_batch_size", 1)), 1)
        image_height = parser.get_int_param("image_height", 640)
        image_width = parser.get_int_param("image_width", 640)
        conf = parser.get_float_param("conf", 0.2)
        iou = parser.get_float_param("iou", 0.2)
        rect = parser.get_bool_param("rect", True)
        max_det = parser.get_int_param("max_det", 300)
        classes = parser.get_list_param("classes")
        
        return YOLOConfig(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            conf=conf,
            iou=iou,
            rect=rect,
            max_det=max_det,
            classes=classes
        )


class ModelFactory:
    @staticmethod
    def create_model(args: Dict[str, Any]) -> YOLO:
        model_path = os.path.join(
            args["model_repository"], 
            args["model_version"], 
            "yolo_int8.engine"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        return YOLO(model_path)


class DeviceManager:
    @staticmethod
    def get_device(device_id: int) -> torch.device:
        return (
            torch.device(f"cuda:{device_id}") 
            if torch.cuda.is_available() 
            else torch.device("cpu")
        )


class ResponseBuilder:
    @staticmethod
    def create_success_response(serialized_results: List[Dict[str, Any]]):
        batch_json = json.dumps(serialized_results, ensure_ascii=False, separators=(',', ':'))
        arr = np.array([batch_json], dtype=np.object_)
        output_tensor = pb_utils.Tensor("result", arr)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    @staticmethod
    def create_error_response(error_message: str):
        return pb_utils.InferenceResponse(
            error=pb_utils.TritonError(error_message)
        )


class TritonPythonModel:
    def __init__(self):
        self._config: Optional[YOLOConfig] = None
        self._model: Optional[YOLO] = None
        self._device: Optional[torch.device] = None
        self._processor = YOLOProcessor()

    def initialize(self, args: Dict[str, Any]) -> None:
        try:
            config_dict = json.loads(args["model_config"])
            self._config = ConfigParser.create_config(config_dict)
            device_id = int(args["model_instance_device_id"])
            
            self._model = ModelFactory.create_model(args)
            self._device = DeviceManager.get_device(device_id)
            
            self._warmup()
            pb_utils.Logger.log_info("YOLO model initialized successfully")
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Initialization failed: {str(e)}")
            raise

    def _warmup(self) -> None:
        dummy = np.zeros((self._config.image_height, self._config.image_width, 3), dtype=np.uint8)
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
                    responses.append(ResponseBuilder.create_error_response("Empty input data"))
                    continue
                
                processed_images = self._processor.preprocess_images(image_data)
                
                results = self._model.predict(
                    source=processed_images,
                    conf=self._config.conf,
                    iou=self._config.iou,
                    imgsz=(self._config.image_height, self._config.image_width),
                    device=self._device,
                    verbose=False,
                    augment=False,
                    rect=self._config.rect,
                    max_det=self._config.max_det,
                    classes=self._config.classes
                )
                
                serialized = self._processor.serialize_results(results)
                response = ResponseBuilder.create_success_response(serialized)
                responses.append(response)
            
            except Exception as e:
                pb_utils.Logger.log_error(f"Request processing failed: {str(e)}")
                responses.append(ResponseBuilder.create_error_response(str(e)))
                
        return responses

    def finalize(self) -> None:
        if self._model is not None:
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pb_utils.Logger.log_info("Model finalized successfully")