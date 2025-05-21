import json
import os
import shutil
import sys
import tempfile
import logging

import numpy as np
import torch
import re
from PIL import Image
from cuda import cudart
from engine import TRTInferenceEngine

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

file_location = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(file_location))
sys.path.insert(0, os.path.join(file_location, "ocr"))

import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    def _set_defaults(self):
        self._batch_size = 1
        self._image_height = 384
        self._image_width = 384
        self._max_length = 64
        self._num_beams = 5
        self._prompt = None
        self._task_start_token = "<s_500k>"
        self._prompt_end_token = "<s_prompt>"

    def _set_from_parameter(self, parameter, parameters, class_):
        value = parameters.get(parameter, None)
        if value is not None:
            value = value["string_value"]
            if value:
                setattr(self, "_" + parameter, class_(value))

    def _set_from_config(self, model_config):
        model_config = json.loads(model_config)
        self._batch_size = int(model_config.get("max_batch_size", 1))
        if self._batch_size < 1:
            self._batch_size = 1

        config_parameters = model_config.get("parameters", {})

        if config_parameters:
            parameter_type_map = {
                "image_height": int,
                "image_width": int,
                "max_length": int,
                "num_beams": int,
                "prompt": str,
                "task_start_token": str,
                "prompt_end_token": str
            }

            for parameter, parameter_type in parameter_type_map.items():
                self._set_from_parameter(parameter, config_parameters, parameter_type)

    def initialize(self, args):
        self._set_defaults()
        self._set_from_config(args["model_config"])
        self._model_instance_device_id = int(args["model_instance_device_id"])
        
        try:            
            model_directory = os.path.join(args["model_repository"], args["model_version"])
            model_path = os.path.join(model_directory, "donut_fp16.pt")
            processor_path = os.path.join(model_directory, "checkpoint")
            
            if not os.path.exists(model_path) or not os.path.exists(processor_path):
                raise Exception(f"Model files not found at {model_path} or {processor_path}")
            
            if torch.cuda.is_available():
                self._device = torch.device(f"cuda:{self._model_instance_device_id}")
            else:
                self._device = torch.device("cpu")
            
            self._engine = TRTInferenceEngine(
                model_path=model_path,
                processor_path=processor_path,
                device=self._device,
                image_size=(self._image_width, self._image_height),
                max_length=self._max_length,
                num_beams=self._num_beams,
                task_start_token=self._task_start_token,
                prompt_end_token=self._prompt_end_token
            )
            
            dummy_image = np.zeros((self._image_height, self._image_width, 3), dtype=np.uint8)
            dummy_pil_image = Image.fromarray(dummy_image)
            
            self._engine.process_image(
                image=dummy_pil_image,
                max_length=self._max_length,
                prompt=self._prompt,
                return_json=True
            )
            
            self._logger = pb_utils.Logger
            self._logger.log_info("Donut model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Donut model: {str(e)}")
            raise

    def execute(self, requests):
        responses = []
        
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image_np = image_tensor.as_numpy()
                pil_images = []
                for img in image_np:
                    pil_images.append(Image.fromarray(img))
                
                try:
                    batch_results = self._engine.process_batch(
                        images=pil_images,
                        batch_size=1, # При компиляции в tensorrt учесть этот параметр
                        max_length=self._max_length,
                        prompt=self._prompt,
                        return_json=True
                    )

                    result_json = json.dumps(batch_results)
                    logger.info(f"Output result_json: {result_json}")
                    result_array = np.array([result_json], dtype=np.object_)
                    logger.info(f"Output result_array: {result_array}, shape: {result_array.shape}, dtype: {result_array.dtype}")
                    result_tensor = pb_utils.Tensor(
                        "text_sequence",
                        result_array
                    )
                    logger.info(f"Output result_tensor: {result_tensor}")
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[result_tensor]
                    )
                    responses.append(inference_response)
                    
                except Exception as process_error:
                    error_message = f"Error during Donut processing: {str(process_error)}"
                    logger.error(error_message, exc_info=True)
                    responses.append(self._create_error_response(error_message))
                    
            except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                logger.error(error_message, exc_info=True)
                responses.append(self._create_error_response(error_message))
                
        return responses
        
    def _create_error_response(self, error_message):
        error_text = [f"ERROR: {error_message}"]
        error_tensor = pb_utils.Tensor("text_sequence", np.array(error_text, dtype=np.object_))
        return pb_utils.InferenceResponse(output_tensors=[error_tensor])

    def finalize(self):
        self._model = None
        self._engine = None
        torch.cuda.empty_cache()
