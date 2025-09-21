import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from engine import DonutInferenceTRT
import triton_python_backend_utils as pb_utils
from dataclasses import dataclass
from transformers import DonutProcessor as HFDonutProcessor
import io


@dataclass
class DonutConfig:
    batch_size: int = 1
    image_height: int = 384
    image_width: int = 384
    max_length: int = 64
    num_beams: int = 5
    prompt: Optional[str] = None
    task_start_token: str = "<s_500k>"
    prompt_end_token: str = "<s_prompt>"


class DonutProcessor:
    @staticmethod
    def serialize_results(results: Any) -> List[Dict[str, Any]]:
        if isinstance(results, list):
            return results
        return [{"text": str(results)}]

    @staticmethod
    def preprocess_images(image_data: np.ndarray) -> List[Image.Image]:
        return [Image.fromarray(image_data[i]) for i in range(image_data.shape[0])]


class ParameterParser:
    def __init__(self, params: Dict[str, str]):
        self._params = params

    def get_param(self, key: str, default: str = "") -> str:
        return self._params.get(key, default)

    def get_int_param(self, key: str, default: int) -> int:
        return int(self.get_param(key, str(default)))

    def get_optional_param(self, key: str) -> Optional[str]:
        value = self.get_param(key)
        return value if value.strip() else None


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
    def create_config(config_dict: Dict[str, Any]) -> DonutConfig:
        params = ConfigParser.parse_parameters(config_dict)
        parser = ParameterParser(params)
        
        batch_size = max(int(config_dict.get("max_batch_size", 1)), 1)
        image_height = parser.get_int_param("image_height", 384)
        image_width = parser.get_int_param("image_width", 384)
        max_length = parser.get_int_param("max_length", 64)
        num_beams = parser.get_int_param("num_beams", 5)
        prompt = parser.get_optional_param("prompt")
        task_start_token = parser.get_param("task_start_token", "<s_500k>")
        prompt_end_token = parser.get_param("prompt_end_token", "<s_prompt>")
        
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


class EngineFactory:
    @staticmethod
    def create_engine(args: Dict[str, Any], config: DonutConfig, device: torch.device) -> DonutInferenceTRT:
        model_directory = os.path.join(args["model_repository"], args["model_version"])
        tensorrt_dir = os.path.join(model_directory, "donut", "engine")
        model_path = os.path.join(model_directory, "donut")
        
        if not os.path.exists(tensorrt_dir):
            raise FileNotFoundError(f"Model files not found at {tensorrt_dir}")

        processor = HFDonutProcessor.from_pretrained(model_path, use_fast=True)
        processor.image_processor.size = (config.image_height, config.image_width)[::-1]
        processor.image_processor.do_align_long_axis = False
        
        return (DonutInferenceTRT(
            tensorrt_dir=tensorrt_dir,
            device=device,
            batch_size=config.batch_size
        ), processor)


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
        batch_jsons = [
            json.dumps(res, ensure_ascii=False, separators=(',', ':')) 
            for res in serialized_results
        ]
        arr = np.array(batch_jsons, dtype=np.object_)
        output_tensor = pb_utils.Tensor("text_sequence", arr)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    @staticmethod
    def create_error_response(error_message: str):
        return pb_utils.InferenceResponse(
            error=pb_utils.TritonError(error_message)
        )


class TritonPythonModel:
    def __init__(self):
        self._config: Optional[DonutConfig] = None
        self._engine: Optional[DonutInferenceTRT] = None
        self._device: Optional[torch.device] = None
        self._processor = DonutProcessor()

    def initialize(self, args: Dict[str, Any]) -> None:
        try:
            config_dict = json.loads(args["model_config"])
            self._config = ConfigParser.create_config(config_dict)
            device_id = int(args["model_instance_device_id"])
            
            self._device = DeviceManager.get_device(device_id)
            self._engine, self.processor = EngineFactory.create_engine(args, self._config, self._device)
            
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

        pixel_values = self.processor(
            dummy_pil_image, 
            return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.squeeze().unsqueeze(0)
        
        self._engine.predict_batch(pixel_values)

    def execute(self, requests) -> List:
        responses = []
        
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image_data = image_tensor.as_numpy()
                
                if image_data.size == 0:
                    raise ValueError("Empty input data")

                list_img = self._processor.preprocess_images(image_data)

                pixel_values = self.processor(
                    list_img, 
                    return_tensors="pt"
                ).pixel_values
                
                batch_results = self._engine.predict_batch(pixel_values)
                
                serialized = self._processor.serialize_results(batch_results)
                response = ResponseBuilder.create_success_response(serialized)
                responses.append(response)
                
            except Exception as e:
                pb_utils.Logger.log_error(f"Request processing failed: {str(e)}")
                responses.append(ResponseBuilder.create_error_response(str(e)))
                
        return responses

    def finalize(self) -> None:
        if self._engine is not None:
            self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pb_utils.Logger.log_info("Model finalized successfully")