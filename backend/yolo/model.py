import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torchvision
import cv2
from cuda import cudart
from PIL import Image
import logging
from ultralytics.engine.results import Results
from ultralytics import YOLO

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


class TritonPythonModel:
    def _set_defaults(self):
        self._batch_size = 16
        self._image_height = 640
        self._image_width = 640

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
                "image_width": int
            }

            for parameter, parameter_type in parameter_type_map.items():
                self._set_from_parameter(parameter, config_parameters, parameter_type)

    def initialize(self, args):
        self._set_defaults()
        self._set_from_config(args["model_config"])
        self._model_instance_device_id = int(args["model_instance_device_id"])
        
        try:
            model_directory = os.path.join(args["model_repository"], args["model_version"])
            model_path = os.path.join(model_directory, "yolo_int8.engine")
    
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
                
            self._model = YOLO(model_path)
            
            if torch.cuda.is_available():
                self._device = torch.device(f"cuda:{self._model_instance_device_id}")
            else:
                self._device = torch.device("cpu")
                
            dummy_image = np.zeros((self._image_height, self._image_width, 3), dtype=np.uint8)
            self._model.predict(dummy_image)
            
            self._logger = pb_utils.Logger
            self._logger.log_info("YOLO model initialized successfully")
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Error initializing YOLO model: {str(e)}")
            raise

    def execute(self, requests):
        responses = []
        
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image_data = input_tensor.as_numpy()
                image_tensor = torch.from_numpy(
                    np.transpose(image_data.astype(np.float32) / 255.0, (0, 3, 1, 2))
                ).to(self._device)
                    
                try:                    
                    batch_results = self._model.predict(
                        source=image_tensor,
                        conf=0.2,
                        iou=0.2,
                        imgsz=(self._image_height, self._image_width),
                        device=self._device,
                        verbose=False,
                        augment=False
                    )
                    logger.warning(batch_results)
                    
                    result_json = json.dumps(batch_results)
                    logger.info(f"Output result_json: {result_json}")
                    result_array = np.array([result_json], dtype=np.object_)
                    logger.info(f"Output result_array: {result_array}, shape: {result_array.shape}, dtype: {result_array.dtype}")
                    result_tensor = pb_utils.Tensor(
                        "result",
                        result_array
                    )
                    logger.info(f"Output result_tensor: {result_tensor}")
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[result_tensor]
                    )
                    responses.append(inference_response)

                except Exception as inference_error:
                    error_message = f"Error during batch YOLO inference: {str(inference_error)}"
                    self._logger.log_error(error_message)
                    responses.append(self._create_error_response(error_message))
                    
            except Exception as request_error:
                error_message = f"Error processing request: {str(request_error)}"
                responses.append(self._create_error_response(error_message))
                
        return responses
    
    def _create_error_response(self, error_message):
        error_json = json.dumps({"error": error_message})
        error_tensor = pb_utils.Tensor("result", np.array([error_json], dtype=np.object_))
        return pb_utils.InferenceResponse(output_tensors=[error_tensor])

    def finalize(self):
        self._model = None
        torch.cuda.empty_cache()