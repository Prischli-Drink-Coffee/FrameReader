import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from engine import TRTInferenceEngine
import triton_python_backend_utils as pb_utils


class DonutConfig:
    def __init__(
        self,
        batch_size: int = 1,
        image_height: int = 384,
        image_width: int = 384,
        max_length: int = 64,
        num_beams: int = 5,
        prompt: Optional[str] = None,
        task_start_token: str = "<s_500k>",
        prompt_end_token: str = "<s_prompt>"
    ):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.max_length = max_length
        self.num_beams = num_beams
        self.prompt = prompt
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token


class DonutProcessor:
    @staticmethod
    def serialize_results(results: Any) -> List[Dict[str, Any]]:
        if isinstance(results, list):
            return results
        return [{"text": str(results)}]

    @staticmethod
    def preprocess_images(image_data: np.ndarray) -> List[Image.Image]:
        return [Image.fromarray(image_data[i]) for i in range(image_data.shape[0])]


class TritonPythonModel:
    def __init__(self):
        self._config: Optional[DonutConfig] = None
        self._engine: Optional[TRTInferenceEngine] = None
        self._device: Optional[torch.device] = None
        self._processor = DonutProcessor()
        self._params: Dict[str, str] = {}

    def _get_param(self, key: str, default: str = "") -> str:
        return self._params.get(key, default)

    def _get_int_param(self, key: str, default: int) -> int:
        return int(self._get_param(key, str(default)))

    def _get_optional_param(self, key: str) -> Optional[str]:
        value = self._get_param(key)
        return value if value.strip() else None

    def _parse_parameters(self, config: Dict[str, Any]) -> None:
        self._params = {}
        for param in config.get("parameters", []):
            key = param["key"]
            value = param["value"]["string_value"]
            self._params[key] = value

    def _create_config(self, config_dict: Dict[str, Any]) -> DonutConfig:
        self._parse_parameters(config_dict)
        
        batch_size = max(int(config_dict.get("max_batch_size", 1)), 1)
        image_height = self._get_int_param("image_height", 384)
        image_width = self._get_int_param("image_width", 384)
        max_length = self._get_int_param("max_length", 64)
        num_beams = self._get_int_param("num_beams", 5)
        prompt = self._get_optional_param("prompt")
        task_start_token = self._get_param("task_start_token", "<s_500k>")
        prompt_end_token = self._get_param("prompt_end_token", "<s_prompt>")
        
        return DonutConfig(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            max_length=max_length,
            num_beams=num_beams,
            prompt=prompt,
            task_start_token=task_start_token,
            prompt_end_token=prompt_end_token
        )

    def initialize(self, args: Dict[str, Any]) -> None:
        try:
            config_dict = json.loads(args["model_config"])
            self._config = self._create_config(config_dict)
            device_id = int(args["model_instance_device_id"])
            
            model_directory = os.path.join(args["model_repository"], args["model_version"])
            model_path = os.path.join(model_directory, "donut_fp16.pt")
            processor_path = os.path.join(model_directory, "checkpoint")
            
            if not os.path.exists(model_path) or not os.path.exists(processor_path):
                raise FileNotFoundError(f"Model files not found at {model_path} or {processor_path}")
            
            self._device = (
                torch.device(f"cuda:{device_id}") 
                if torch.cuda.is_available() 
                else torch.device("cpu")
            )
            
            self._engine = TRTInferenceEngine(
                model_path=model_path,
                processor_path=processor_path,
                device=self._device,
                image_size=(self._config.image_width, self._config.image_height),
                max_length=self._config.max_length,
                num_beams=self._config.num_beams,
                task_start_token=self._config.task_start_token,
                prompt_end_token=self._config.prompt_end_token
            )
            
            self._warmup()
            pb_utils.Logger.log_info("Donut model initialized successfully")
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Initialization failed: {str(e)}")
            raise

    def _warmup(self) -> None:
        dummy_image = np.zeros(
            (self._config.image_height, self._config.image_width, 3), 
            dtype=np.uint8
        )
        dummy_pil_image = Image.fromarray(dummy_image)
        
        self._engine.process_image(
            image=dummy_pil_image,
            max_length=self._config.max_length,
            prompt=self._config.prompt,
            return_json=True
        )

    def execute(self, requests) -> List[pb_utils.InferenceResponse]:
        responses = []
        
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image_data = image_tensor.as_numpy()
                
                if image_data.size == 0:
                    raise ValueError("Empty input data")
                
                batch_results = self._engine.process_batch(
                    images=self._processor.preprocess_images(image_data),
                    batch_size=image_data.shape[0],
                    max_length=self._config.max_length,
                    prompt=self._config.prompt,
                    return_json=True
                )
                
                serialized = self._processor.serialize_results(batch_results)
                batch_jsons = [
                    json.dumps(res, ensure_ascii=False, separators=(',', ':')) 
                    for res in serialized
                ]
                arr = np.array(batch_jsons, dtype=np.object_)
                output_tensor = pb_utils.Tensor("text_sequence", arr)
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
        if self._engine is not None:
            self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pb_utils.Logger.log_info("Model finalized successfully")