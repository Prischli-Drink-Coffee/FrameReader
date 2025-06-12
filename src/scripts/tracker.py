import os
import cv2
import numpy as np
import torch
import torchvision
import time
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator, NamedTuple, AsyncIterator, AsyncGenerator

from ultralytics import YOLO
from ultralytics.trackers import BYTETracker, BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from pathlib import Path
from dataclasses import dataclass, field
import traceback
import asyncio

from src.triton_api.stream_endpoint import StreamEndpointClient
from src.triton_api.websocket_endpoint import WebSocketEndpointClient


from src.utils.custom_logging import setup_logging
# from src.utils.env import Env # Если Env нужен для URL по умолчанию в SAHITrackingWrapper

log = setup_logging()
# env = Env() # Если используется

_colors_list = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (192, 192, 192), (128, 128, 128), (255, 165, 0), (255, 192, 203)
]


@dataclass
class FrameTrackingResult:
    frame_number: int
    timestamp: float
    tracked_objects: List[Dict[str, Any]]
    frame_shape: Tuple[int, int]
    processing_time: float
    annotated_frame: Optional[np.ndarray] = None


def get_color(index: int) -> Tuple[int, int, int]:
    return _colors_list[index % len(_colors_list)]


class MockBoxesData: # Остается без изменений
    def __init__(self, boxes_data_np: np.ndarray, orig_shape: Tuple[int, int]):
        if boxes_data_np.ndim == 1 and boxes_data_np.shape[0] == 6:
            boxes_data_np = boxes_data_np[np.newaxis, :]
        elif boxes_data_np.shape[0] == 0:
            boxes_data_np = np.empty((0, 6), dtype=np.float32)

        self.data = torch.from_numpy(boxes_data_np).float().cpu() # Keep on CPU for consistency
        self.orig_shape = orig_shape

        if self.data.numel() > 0:
            self.xyxy = self.data[:, :4]
            self.conf = self.data[:, 4]
            self.cls = self.data[:, 5].int()
            x1, y1, x2, y2 = self.xyxy.unbind(1)
            self.xywh = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)
        else:
            self.xyxy = torch.empty((0, 4), device=self.data.device)
            self.conf = torch.empty(0, device=self.data.device)
            self.cls = torch.empty(0, dtype=torch.int, device=self.data.device)
            self.xywh = torch.empty((0, 4), device=self.data.device)

    def cpu(self): # Already on CPU
        return self

    def __len__(self):
        return self.data.shape[0]


class MockResults: # Остается без изменений
    def __init__(self, boxes_data_np: np.ndarray, orig_shape: Tuple[int, int]):
        self.boxes = MockBoxesData(boxes_data_np, orig_shape)
        self.conf = self.boxes.conf
        self.xywh = self.boxes.xywh
        self.cls = self.boxes.cls
        self.names = {} # This will be populated by SAHITrackingWrapper
        self.masks = None
        self.probs = None
        self.keypoints = None
        self.orig_shape = orig_shape
        self.orig_img = None # Can be set if needed by tracker

    def cpu(self):
        self.boxes.cpu()
        return self

    def __len__(self):
        return len(self.boxes)


class SAHITrackingWrapper:
    def __init__(
        self,
        model_path: Optional[str] = None,
        tracker_type: str = 'botsort',
        tracker_config_path: Optional[str] = None,
        frame_rate: int = 30,
        window_size_ratio: Tuple[float, float] = (0.7, 0.7),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        img_size: Union[int, Tuple[int,int]] = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45, # For local model NMS before SAHI
        nms_threshold_global: float = 0.5, # NMS after SAHI, before tracker
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
        # Triton integration parameters
        detection_source: str = "local",  # "local", "triton_stream", "triton_ws"
        triton_stream_url: Optional[str] = None,
        triton_ws_url: Optional[str] = None,
        triton_model_name: str = "yolo", # Model name on Triton server
        triton_chunk_size: int = 1,
        class_names_map: Optional[Dict[int, str]] = None # e.g. {0: 'person', 1: 'car'}
    ):
        self.detection_source = detection_source
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.debug(f"Using device for NMS and potentially local model: {self.device}")

        self.model_names: Dict[int, str] = {}
        self.model_names_reverse_map: Dict[str, int] = {}
        
        self.triton_stream_client: Optional[StreamEndpointClient] = None
        self.triton_ws_client: Optional[WebSocketEndpointClient] = None

        if self.detection_source == "local":
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found or not provided for local detection: {model_path}")
            log.debug(f"Loading local detection model from {model_path}...")
            self.model = YOLO(model_path, task='detect')
            # self.model.to(self.device) # Ensure local model is on the correct device
            self.model_names = self.model.names
            log.debug(f"Local detection model loaded. Names: {self.model_names}")
        elif self.detection_source == "triton_stream":
            if not triton_stream_url:
                raise ValueError("Triton stream URL must be provided for triton_stream source.")
            self.triton_stream_client = StreamEndpointClient(base_url=triton_stream_url, chunk_size=triton_chunk_size)
            log.debug(f"Using Triton Stream client with URL: {triton_stream_url}")
            if class_names_map:
                self.model_names = class_names_map
        elif self.detection_source == "triton_ws":
            if not triton_ws_url:
                raise ValueError("Triton WebSocket URL must be provided for triton_ws source.")
            self.triton_ws_client = WebSocketEndpointClient(base_url=triton_ws_url)
            # WebSocket client connection is managed per session in run_inference_session
            log.debug(f"Using Triton WebSocket client with URL: {triton_ws_url}")
            if class_names_map:
                self.model_names = class_names_map
        else:
            raise ValueError(f"Unsupported detection_source: {self.detection_source}")

        if not self.model_names and self.detection_source != "local":
            log.warning("class_names_map not provided for Triton. Class names will be 'Cls_<id>' or based on hash.")
        
        if self.model_names:
            self.model_names_reverse_map = {v: k for k, v in self.model_names.items()}

        self.triton_model_name = triton_model_name
        self.tracker_type = tracker_type

        script_dir = Path(__file__).parent.resolve()
        default_cfg_path = script_dir / 'cfg' / 'trackers' / f'{tracker_type}.yaml'

        if tracker_config_path and os.path.exists(tracker_config_path):
            tracker_config_file = tracker_config_path
        elif default_cfg_path.exists():
            tracker_config_file = str(default_cfg_path)
        else:
            try:
                tracker_config_file = check_yaml(f'{tracker_type}.yaml')
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not find tracker config: '{tracker_type}.yaml'. "
                    f"Provide a valid 'tracker_config_path' or ensure it's in ultralytics defaults."
                ) from e
        
        log.debug(f"Using tracker config: {tracker_config_file}")
        tracker_cfg_dict = YAML.load(str(tracker_config_file))
        if tracker_cfg_dict.get('tracker_type') != tracker_type:
            log.warning(f"Tracker config file specifies type '{tracker_cfg_dict.get('tracker_type')}', "
                        f"but '{tracker_type}' was requested. Overriding to '{tracker_type}'.")
            tracker_cfg_dict['tracker_type'] = tracker_type
        
        tracker_cfg = IterableSimpleNamespace(**tracker_cfg_dict)
        self.tracker_cfg_dict = tracker_cfg_dict # Save for reset
        self.current_frame_rate = frame_rate # Save for reset

        if tracker_cfg.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError(f"Unsupported tracker type: {tracker_cfg.tracker_type}")

        log.info(f"Initializing {tracker_cfg.tracker_type} tracker (frame_rate={frame_rate})...")
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](args=tracker_cfg, frame_rate=frame_rate)

        self.window_size_ratio = window_size_ratio
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size # Can be int or tuple
        self.conf_threshold = conf_threshold # For local model or server-side if configurable
        self.iou_threshold = iou_threshold   # For local model NMS
        self.nms_threshold_global = nms_threshold_global
        self.classes = classes # For local model filtering

    def _get_class_id(self, class_name: str) -> int:
        if not self.model_names_reverse_map:
            log.debug(f"model_names_reverse_map is empty. Mapping class name '{class_name}' to hash-based ID.")
            import hashlib
            return int(hashlib.md5(class_name.encode()).hexdigest(), 16) % 1000 
        
        class_id = self.model_names_reverse_map.get(class_name)
        if class_id is None:
            log.warning(f"Class name '{class_name}' not found in model_names_reverse_map. Using hash-based ID.")
            import hashlib
            return int(hashlib.md5(class_name.encode()).hexdigest(), 16) % 1000
        return class_id

    def _adapt_triton_detections_to_torch(
        self, 
        triton_detections_for_window: List[Dict[str, Any]],
        window_orig_w: int, window_orig_h: int # Original window dimensions for context
    ) -> Dict[str, torch.Tensor]:
        """
        Adapts a list of detections from Triton for a single window into torch tensors.
        Input: [{'box2d': [x1,y1,x2,y2], 'score': float, 'label': str}, ...]
        Output: {'xyxy': tensor, 'conf': tensor, 'cls': tensor}
        Coordinates are assumed to be relative to the window.
        """
        xyxys, confs, clss = [], [], []
        if not triton_detections_for_window:
            return {
                'xyxy': torch.empty((0, 4), device=self.device, dtype=torch.float32),
                'conf': torch.empty(0, device=self.device, dtype=torch.float32),
                'cls': torch.empty(0, device=self.device, dtype=torch.int32)
            }

        for det in triton_detections_for_window:
            box = det.get('box2d') # Expected: [x1, y1, x2, y2] relative to window
            score = det.get('score')
            label = det.get('label')

            if box is None or score is None or label is None:
                log.warning(f"Skipping malformed Triton detection: {det}")
                continue
            
            # Ensure box coordinates are within window boundaries (optional, depends on server output)
            # x1, y1, x2, y2 = box
            # x1 = max(0, min(x1, window_orig_w))
            # y1 = max(0, min(y1, window_orig_h))
            # x2 = max(0, min(x2, window_orig_w))
            # y2 = max(0, min(y2, window_orig_h))
            # if x1 >= x2 or y1 >= y2:
            #     log.debug(f"Skipping invalid box {box} after clamping for window {window_orig_w}x{window_orig_h}")
            #     continue
            # xyxys.append([x1,y1,x2,y2])

            xyxys.append(box)
            confs.append(score)
            clss.append(self._get_class_id(label))
        
        return {
            'xyxy': torch.tensor(xyxys, device=self.device, dtype=torch.float32) if xyxys else torch.empty((0, 4), device=self.device, dtype=torch.float32),
            'conf': torch.tensor(confs, device=self.device, dtype=torch.float32) if confs else torch.empty(0, device=self.device, dtype=torch.float32),
            'cls': torch.tensor(clss, device=self.device, dtype=torch.int32) if clss else torch.empty(0, device=self.device, dtype=torch.int32)
        }

    def _adapt_triton_results(
        self, 
        triton_payload: List[Dict[str, Any]], # List of {'image_id': str, 'detections': List[DetDict]} or List of {'detections': List[DetDict]}
        num_expected_windows: int,
        window_coords_list: List[Tuple[int,int,int,int]] # For original window dimensions
    ) -> List[Dict[str, torch.Tensor]]:
        """
        General adapter for Triton results (Stream or WS).
        Assumes triton_payload is a list where each item corresponds to a window.
        Each item is a dict, hopefully containing 'detections' and optionally 'image_id'.
        If 'image_id' is present and reliable, it could be used for robust ordering.
        """
        adapted_results = []
        
        if not triton_payload and num_expected_windows > 0:
            log.warning("Triton returned empty payload but windows were expected.")
            for i in range(num_expected_windows):
                 adapted_results.append(self._adapt_triton_detections_to_torch([], 0,0)) # Empty detections
            return adapted_results

        if len(triton_payload) != num_expected_windows:
            log.warning(f"Triton results count ({len(triton_payload)}) mismatch with expected windows ({num_expected_windows}). Adapting based on received.")
            # Potentially pad or truncate, or rely on image_id if available and robust.
            # For now, process what's received. This might lead to issues if order is lost.

        for i, window_result_data in enumerate(triton_payload):
            if i >= num_expected_windows: # More results than windows, shouldn't happen if 1-to-1
                break 
            
            win_x1, win_y1, win_x2, win_y2 = window_coords_list[i]
            win_w, win_h = win_x2 - win_x1, win_y2 - win_y1

            # 'detections' is the key assumed from StreamEndpointClient and patched WebSocketEndpointClient
            detections_for_this_window = window_result_data.get('detections', []) 
            adapted_results.append(
                self._adapt_triton_detections_to_torch(detections_for_this_window, win_w, win_h)
            )
        
        # If Triton returned fewer results than expected, pad with empty detections
        while len(adapted_results) < num_expected_windows:
            log.debug(f"Padding missing Triton result for window index {len(adapted_results)}")
            adapted_results.append(self._adapt_triton_detections_to_torch([],0,0))

        return adapted_results


    def _get_windows(self, frame_width: int, frame_height: int) -> List[Tuple[int, int, int, int]]:
        # SAHI slicing logic (remains synchronous)
        win_w_float = frame_width * self.window_size_ratio[0]
        win_h_float = frame_height * self.window_size_ratio[1]

        # Ensure window dimensions are at least 1, prevent zero division
        win_w = max(1, int(win_w_float))
        win_h = max(1, int(win_h_float))
        
        overlap_w = int(win_w * self.overlap_ratio[0])
        overlap_h = int(win_h * self.overlap_ratio[1])

        step_w = max(1, win_w - overlap_w)
        step_h = max(1, win_h - overlap_h)

        windows = []
        for y_start in range(0, frame_height, step_h):
            y_end_candidate = y_start + win_h
            # Adjust y1 if window goes out of bounds (take from end)
            y1 = y_start if y_end_candidate <= frame_height else max(0, frame_height - win_h)
            y2 = min(y1 + win_h, frame_height) # Ensure y2 does not exceed frame_height

            if y1 >= y2 : continue # Skip if window has zero or negative height

            for x_start in range(0, frame_width, step_w):
                x_end_candidate = x_start + win_w
                # Adjust x1 if window goes out of bounds
                x1 = x_start if x_end_candidate <= frame_width else max(0, frame_width - win_w)
                x2 = min(x1 + win_w, frame_width) # Ensure x2 does not exceed frame_width
                
                if x1 >= x2 : continue # Skip if window has zero or negative width

                window_coords = (x1, y1, x2, y2)
                # Add if valid and not already added (though range logic should prevent duplicates)
                if window_coords not in windows:
                     windows.append(window_coords)

                if x2 == frame_width: break # Reached end of row
            if y2 == frame_height: break # Reached end of columns
        
        if not windows and frame_width > 0 and frame_height > 0: # Ensure at least one window for valid frames
            log.warning("SAHI _get_windows returned no windows. Adding full frame as a window.")
            windows.append((0,0,frame_width, frame_height))
        elif not windows:
            log.warning("SAHI _get_windows returned no windows for zero-dim frame.")


        return windows

    def _nms_global(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        # Global NMS logic (remains synchronous, uses self.device)
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        detection_list = []
        for det in detections:
            xyxy_abs = det['xyxy'] # Absolute frame coordinates
            detection_list.append([xyxy_abs[0], xyxy_abs[1], xyxy_abs[2], xyxy_abs[3], 
                                   det['confidence'], det['class_id']])
        
        # Ensure NMS runs on the configured device (CPU or GPU)
        detections_tensor = torch.tensor(detection_list, dtype=torch.float32, device=self.device)

        if detections_tensor.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = detections_tensor[:, :4]
        scores = detections_tensor[:, 4]
        classes = detections_tensor[:, 5] # class_id already int

        final_detections_list = []
        unique_classes = torch.unique(classes)

        for cls_id_tensor in unique_classes:
            cls_id = cls_id_tensor.item()
            cls_mask = (classes == cls_id)
            if not torch.any(cls_mask): continue

            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            keep_indices_cls = torchvision.ops.nms(cls_boxes, cls_scores, self.nms_threshold_global)
            
            # NMS returns indices relative to the input of NMS (cls_boxes)
            # We need to map these back to indices in detections_tensor
            original_indices_for_class = torch.where(cls_mask)[0]
            final_detections_list.append(detections_tensor[original_indices_for_class[keep_indices_cls]])

        if not final_detections_list:
            return np.empty((0, 6), dtype=np.float32)

        merged_detections_tensor = torch.cat(final_detections_list, dim=0)
        return merged_detections_tensor.cpu().numpy() # Tracker expects numpy

    async def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        frame_height, frame_width = frame.shape[:2]
        windows = self._get_windows(frame_width, frame_height)
        all_detections_in_frame_coords = []
        
        if not windows:
            log.debug("No windows generated by SAHI for this frame.")
            return []
            
        batch_images_for_model = [] # Images resized for model input
        window_coords_list = [] # Original coordinates of these windows
        
        # Prepare target_size_w, target_size_h from self.img_size
        if isinstance(self.img_size, int):
            target_size_w, target_size_h = self.img_size, self.img_size
        else: # Tuple
            target_size_w, target_size_h = self.img_size[0], self.img_size[1]

        for x1_w, y1_w, x2_w, y2_w in windows:
            window_img_orig_slice = frame[y1_w:y2_w, x1_w:x2_w]
            if window_img_orig_slice.size == 0:
                log.warning(f"Skipping empty window slice at ({x1_w},{y1_w})-({x2_w},{y2_w})")
                continue
            
            window_coords_list.append((x1_w, y1_w, x2_w, y2_w)) # Store original slice coords
            
            # Resize for model
            if target_size_w != window_img_orig_slice.shape[1] or target_size_h != window_img_orig_slice.shape[0]:
                resized_window_img = cv2.resize(window_img_orig_slice, (target_size_w, target_size_h))
            else:
                resized_window_img = window_img_orig_slice.copy() # Use copy to avoid issues if array is modified
            batch_images_for_model.append(resized_window_img)

        num_real_windows = len(batch_images_for_model)
        if num_real_windows == 0:
            return []
        
        # This will hold detection results for each window,
        # where each item is a dict {'xyxy': tensor, 'conf': tensor, 'cls': tensor}
        # with coordinates relative to the *resized* window_img that went into the model.
        per_window_model_outputs: List[Dict[str, torch.Tensor]] = []

        try:
            if self.detection_source == "local":
                # Ultralytics YOLO.predict can take a list of images
                # Results are also a list, one for each input image
                local_model_results = self.model.predict(
                    source=batch_images_for_model,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold, # NMS for local model's own output per slice
                    imgsz=(target_size_h, target_size_w), # Ensure correct order for imgsz
                    classes=self.classes,
                    device=self.device,
                    verbose=False,
                    augment=False
                )
                for res_obj in local_model_results:
                    # res_obj.boxes contains xyxy, conf, cls tensors
                    per_window_model_outputs.append({
                        'xyxy': res_obj.boxes.xyxy.to(self.device), # Ensure on correct device for consistency
                        'conf': res_obj.boxes.conf.to(self.device),
                        'cls': res_obj.boxes.cls.to(self.device).int()
                    })

            elif self.detection_source == "triton_stream" and self.triton_stream_client:
                async with self.triton_stream_client as client:
                    triton_results_payload = await client.stream_collect_from_arrays(
                        image_arrays=batch_images_for_model,
                        model_name=self.triton_model_name,
                        # chunk_size can be managed by client's default or passed here
                    )
                    # triton_results_payload['results'] is List[Dict{'image_id':str, 'detections':List[DetDict]}]
                    # Adapt this to per_window_model_outputs
                    per_window_model_outputs = self._adapt_triton_results(
                        triton_results_payload.get('results', []), 
                        num_real_windows,
                        window_coords_list # Pass original window coords for context if adapter needs them
                    )
            
            elif self.detection_source == "triton_ws" and self.triton_ws_client:
                 # WebSocket client's run_inference_session handles connect/disconnect per call
                 # or use a managed connection if __aenter__/__aexit__ were to handle connect(model_name)
                triton_ws_payload = await self.triton_ws_client.run_inference_session(
                    model_name=self.triton_model_name,
                    images=batch_images_for_model, # type: ignore
                    # chunk_size can be passed if server supports it for WS stream
                )
                # triton_ws_payload is List[Dict{'detections':List[DetDict]}] (assuming order is preserved)
                per_window_model_outputs = self._adapt_triton_results(
                    triton_ws_payload, 
                    num_real_windows,
                    window_coords_list
                )

            # Post-process detections from per_window_model_outputs
            for i in range(num_real_windows):
                if i >= len(per_window_model_outputs): continue # Should not happen if padding works

                model_output_for_window = per_window_model_outputs[i]
                boxes_xyxy_resized = model_output_for_window['xyxy'] # Coords relative to resized window
                confs_resized = model_output_for_window['conf']
                clss_resized = model_output_for_window['cls']

                if boxes_xyxy_resized.numel() == 0:
                    continue

                win_x1, win_y1, win_x2, win_y2 = window_coords_list[i]
                orig_win_w, orig_win_h = win_x2 - win_x1, win_y2 - win_y1

                for j in range(boxes_xyxy_resized.shape[0]):
                    w_box_resized = boxes_xyxy_resized[j].tolist() # [x1r, y1r, x2r, y2r]
                    conf = confs_resized[j].item()
                    cls_id = clss_resized[j].item() # Already int

                    # Scale box from resized_window (target_size_w, target_size_h) to original_window_slice
                    f_x1 = (w_box_resized[0] / target_size_w) * orig_win_w + win_x1
                    f_y1 = (w_box_resized[1] / target_size_h) * orig_win_h + win_y1
                    f_x2 = (w_box_resized[2] / target_size_w) * orig_win_w + win_x1
                    f_y2 = (w_box_resized[3] / target_size_h) * orig_win_h + win_y1
                    
                    # Clip to frame boundaries
                    f_x1, f_y1 = max(0.0, f_x1), max(0.0, f_y1)
                    f_x2, f_y2 = min(float(frame_width), f_x2), min(float(frame_height), f_y2)
                    
                    if f_x2 > f_x1 and f_y2 > f_y1: # Valid detection
                        all_detections_in_frame_coords.append({
                            "bbox": [f_x1, f_y1, f_x2 - f_x1, f_y2 - f_y1], # x,y,w,h
                            "confidence": conf,
                            "class_id": cls_id, # int
                            "xyxy": [f_x1, f_y1, f_x2, f_y2] # For NMS global
                        })
        except Exception as e:
            log.error(f"Error during detection processing ({self.detection_source}): {type(e).__name__}: {str(e)}")
            log.error(f"Full traceback:\n{traceback.format_exc()}")
            # Fallback to empty if error, or re-raise depending on desired robustness
            return []


        merged_detections_np = self._nms_global(all_detections_in_frame_coords)
        
        # Prepare for tracker
        # MockResults expects numpy array [N, 6] where cols are x1,y1,x2,y2,conf,cls
        mock_results_for_tracker = MockResults(merged_detections_np, (frame_height, frame_width))
        mock_results_for_tracker.names = self.model_names # Pass names to MockResults for tracker use

        try:
            # Tracker update is synchronous
            tracked_output_np = self.tracker.update(mock_results_for_tracker, frame)
        except Exception as e:
            # log.error(f"Error during tracker.update: {type(e).__name__}: {str(e)}")
            # log.error(f"Full traceback for tracker.update error:\n{traceback.format_exc()}")
            return []

        tracked_objects_list = []
        if isinstance(tracked_output_np, np.ndarray) and tracked_output_np.size > 0:
            output_cols = tracked_output_np.shape[1]
            # Columns: x1, y1, x2, y2, track_id, conf, cls_id, [optional_idx], [optional_angle]
            for row in tracked_output_np:
                if output_cols < 7: continue # Minimum expected columns
                x1, y1, x2, y2, track_id, conf, cls_id = row[:7]
                
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id, cls_id = int(track_id), int(cls_id)

                # Ensure box is within frame (tracker might output slightly outside)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)

                if x2 <= x1 or y2 <= y1: continue # Invalid box after clamping

                class_name = self.model_names.get(cls_id, f"Cls_{cls_id}")
                obj_data = {
                    'box': [x1, y1, x2, y2],
                    'track_id': track_id,
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': float(conf)
                }
                if output_cols >= 8 : # original_det_idx (BoTSORT with idx) or angle (OBB)
                    # This logic needs to be specific to the tracker output format
                    # For BoTSORT with idx:
                    if self.tracker_type == 'botsort' and 'idx' in self.tracker.args.get('tracker_cfg',{}).get('public_vars',[]):
                         obj_data['original_det_idx'] = int(row[7])
                    # Add other specific cases if needed, e.g. for OBB angle
                
                tracked_objects_list.append(obj_data)
        elif isinstance(tracked_output_np, (np.ndarray, list)) and len(tracked_output_np) == 0:
            pass # No objects tracked
        else:
            log.warning(f"Tracker returned unexpected output type: {type(tracked_output_np)}")

        return tracked_objects_list

    def annotate_frame( # Remains synchronous
        self, 
        frame: np.ndarray, 
        tracked_objects: List[Dict[str, Any]], 
        show_labels: bool = True, 
        line_width: int = 2
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for obj in tracked_objects:
            box, track_id = obj['box'], obj['track_id']
            class_name, confidence = obj['class_name'], obj['confidence']
            x1, y1, x2, y2 = map(int, box)
            color = get_color(track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_width)
            if show_labels:
                label_text = f"ID:{track_id} {class_name} ({confidence:.2f})"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # Ensure label background is within frame
                label_bg_y1 = max(y1 - text_h - baseline - 3, 0)
                label_bg_y2 = y1 -3 # Should be max(y1 - 3, text_h + baseline) but y1-3 ensures it's above box
                
                # Ensure text itself is visible
                text_y_pos = max(y1 - baseline - 3, text_h) # if box is at top, text_y_pos should be positive

                cv2.rectangle(annotated_frame, (x1, label_bg_y1), (x1 + text_w, label_bg_y2), color, -1)
                cv2.putText(annotated_frame, label_text, (x1, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated_frame

    def reset_tracker(self): # Remains synchronous
        log.info("Resetting tracker state...")
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        elif hasattr(self, 'tracker_cfg_dict') and hasattr(self, 'current_frame_rate'):
            log.warning("Tracker does not have a .reset() method. Re-initializing tracker.")
            tracker_cfg = IterableSimpleNamespace(**self.tracker_cfg_dict)
            TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
            self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](args=tracker_cfg, frame_rate=self.current_frame_rate)
        else:
            log.error("Cannot reset tracker: No .reset() method and insufficient info to re-initialize.")


class VideoStreamTracker:
    def __init__(
        self,
        # Local model params (optional if Triton used)
        model_path: Optional[str] = None,
        # SAHI & Tracker params
        tracker_type: str = 'botsort',
        tracker_config_path: Optional[str] = None,
        window_size_ratio: Tuple[float, float] = (0.7, 0.7),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        img_size: Union[int, Tuple[int,int]] = 640,
        conf: float = 0.25, # Local model conf or general SAHI conf
        iou: float = 0.45,  # Local model NMS iou
        nms_global: float = 0.5, # SAHI global NMS iou
        classes: Optional[List[int]] = None, # Local model classes
        device: Optional[Union[str, torch.device]] = None,
        # Triton params
        detection_source: str = "local",
        triton_stream_url: Optional[str] = None,
        triton_ws_url: Optional[str] = None,
        triton_model_name: str = "yolo",
        triton_chunk_size: int = 1,
        class_names_map: Optional[Dict[int, str]] = None
    ):
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.tracker_config_path = tracker_config_path
        self.window_size_ratio = window_size_ratio
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.nms_global = nms_global
        self.classes = classes
        self.device = device
        
        self.detection_source = detection_source
        self.triton_stream_url = triton_stream_url
        self.triton_ws_url = triton_ws_url
        self.triton_model_name = triton_model_name
        self.triton_chunk_size = triton_chunk_size
        self.class_names_map = class_names_map
        
        self.tracker_wrapper: Optional[SAHITrackingWrapper] = None
        self.current_frame_rate_stored = 30 # Default, updated on stream start

    def _initialize_tracker(self, fps: int): # Synchronous init of wrapper
        self.current_frame_rate_stored = fps
        if self.tracker_wrapper is None:
            log.info(f"Initializing SAHITrackingWrapper with source: {self.detection_source}")
            self.tracker_wrapper = SAHITrackingWrapper(
                model_path=self.model_path,
                tracker_type=self.tracker_type,
                tracker_config_path=self.tracker_config_path,
                frame_rate=fps,
                window_size_ratio=self.window_size_ratio,
                overlap_ratio=self.overlap_ratio,
                img_size=self.img_size,
                conf_threshold=self.conf,
                iou_threshold=self.iou,
                nms_threshold_global=self.nms_global,
                classes=self.classes,
                device=self.device,
                detection_source=self.detection_source,
                triton_stream_url=self.triton_stream_url,
                triton_ws_url=self.triton_ws_url,
                triton_model_name=self.triton_model_name,
                triton_chunk_size=self.triton_chunk_size,
                class_names_map=self.class_names_map
            )

    async def stream_video_tracking(
        self,
        video_path: str,
        vid_stride: int = 1,
        include_annotated_frame: bool = False,
        show_labels: bool = True,
        line_width: int = 2,
        fallback_frame_rate: int = 30
    ) -> AsyncIterator[FrameTrackingResult]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        frame_height, frame_width = -1, -1
        try:
            frame_width_prop = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height_prop = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_width_prop <= 0 or frame_height_prop <= 0:
                ret_test, test_frame = cap.read()
                if ret_test and test_frame is not None:
                    frame_height, frame_width = test_frame.shape[:2]
                    log.info(f"Read frame dimensions from first frame: {frame_width}x{frame_height}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start
                else:
                    raise ValueError("Could not read valid frame dimensions from video.")
            else:
                frame_width, frame_height = frame_width_prop, frame_height_prop

            effective_fps = int(round(native_fps)) if native_fps and native_fps > 0 else fallback_frame_rate
            effective_fps = max(1, effective_fps)

            self._initialize_tracker(effective_fps) # Synchronous call

            frame_num = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success or frame is None:
                    log.info("End of video stream or failed to read frame.")
                    break

                frame_num += 1
                if vid_stride > 1 and (frame_num - 1) % vid_stride != 0:
                    continue

                frame_process_start_time = time.time()
                
                if self.tracker_wrapper is None: # Should not happen if _initialize_tracker worked
                    raise RuntimeError("Tracker wrapper not initialized.")
                
                # process_frame is now async
                tracked_objects = await self.tracker_wrapper.process_frame(frame)
                
                frame_processing_time = time.time() - frame_process_start_time
                timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = timestamp_msec / 1000.0 if timestamp_msec >= 0 else frame_num / effective_fps

                annotated_frame_data = None
                if include_annotated_frame:
                    annotated_frame_data = self.tracker_wrapper.annotate_frame(
                        frame, tracked_objects, show_labels, line_width
                    )

                yield FrameTrackingResult(
                    frame_number=frame_num, timestamp=timestamp,
                    tracked_objects=tracked_objects, frame_shape=(frame_height, frame_width),
                    processing_time=frame_processing_time, annotated_frame=annotated_frame_data
                )
        finally:
            cap.release()

    async def stream_camera_tracking(
        self,
        camera_index: Union[int, str] = 0, # Allow string for e.g. RTSP
        include_annotated_frame: bool = False,
        show_labels: bool = True,
        line_width: int = 2,
        frame_rate: int = 30 # Target FPS for camera
    ) -> AsyncIterator[FrameTrackingResult]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Could not open camera with index/path: {camera_index}")

        cam_frame_height, cam_frame_width = -1,-1
        try:
            # Attempt to set FPS for cameras that support it
            cap.set(cv2.CAP_PROP_FPS, float(frame_rate)) 
            
            ret_test, test_frame = cap.read()
            if not ret_test or test_frame is None:
                raise IOError(f"Could not read initial frame from camera {camera_index}")
            cam_frame_height, cam_frame_width = test_frame.shape[:2]
            
            self._initialize_tracker(frame_rate) # Use target frame_rate

            frame_num = 0
            overall_start_time = time.time()

            while cap.isOpened():
                success, frame = cap.read()
                if not success or frame is None:
                    log.info("Camera stream ended or failed to read frame.")
                    break
                
                current_frame_h, current_frame_w = frame.shape[:2]
                if current_frame_h != cam_frame_height or current_frame_w != cam_frame_width:
                    log.warning(f"Camera resolution changed mid-stream to {current_frame_w}x{current_frame_h}")
                    cam_frame_height, cam_frame_width = current_frame_h, current_frame_w

                frame_num += 1
                frame_process_start_time = time.time()
                
                if self.tracker_wrapper is None:
                    raise RuntimeError("Tracker wrapper not initialized.")
                
                tracked_objects = await self.tracker_wrapper.process_frame(frame)
                
                frame_processing_time = time.time() - frame_process_start_time
                timestamp = time.time() - overall_start_time

                annotated_frame_data = None
                if include_annotated_frame:
                    annotated_frame_data = self.tracker_wrapper.annotate_frame(
                        frame, tracked_objects, show_labels, line_width
                    )

                yield FrameTrackingResult(
                    frame_number=frame_num, timestamp=timestamp,
                    tracked_objects=tracked_objects, frame_shape=(cam_frame_height, cam_frame_width),
                    processing_time=frame_processing_time, annotated_frame=annotated_frame_data
                )
        finally:
            cap.release()

    def reset_tracker(self): # Synchronous
        if self.tracker_wrapper:
            self.tracker_wrapper.reset_tracker()


async def track_video_sahi(
    video_path: str,
    output_path: str,
    # Local model params (optional)
    model_path: Optional[str] = None,
    # SAHI & Tracker params
    tracker_type: str = 'botsort',
    tracker_config_path: Optional[str] = None,
    fallback_frame_rate: int = 30,
    window_size_ratio: Tuple[float, float] = (0.7, 0.7),
    overlap_ratio: Tuple[float, float] = (0.2, 0.2),
    img_size: Union[int, Tuple[int,int]] = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    nms_global: float = 0.5,
    show_labels: bool = True,
    line_width: int = 2,
    show_preview: bool = False,
    vid_stride: int = 1,
    classes: Optional[List[int]] = None,
    output_fps_override: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    # Triton params
    detection_source: str = "local",
    triton_stream_url: Optional[str] = None,
    triton_ws_url: Optional[str] = None,
    triton_model_name: str = "yolo",
    triton_chunk_size: int = 1,
    class_names_map: Optional[Dict[int, str]] = None
) -> Optional[str]:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    stream_tracker = VideoStreamTracker(
        model_path=model_path, tracker_type=tracker_type, tracker_config_path=tracker_config_path,
        window_size_ratio=window_size_ratio, overlap_ratio=overlap_ratio, img_size=img_size,
        conf=conf, iou=iou, nms_global=nms_global, classes=classes, device=device,
        detection_source=detection_source, triton_stream_url=triton_stream_url,
        triton_ws_url=triton_ws_url, triton_model_name=triton_model_name,
        triton_chunk_size=triton_chunk_size, class_names_map=class_names_map
    )

    cap_check = cv2.VideoCapture(video_path) # For properties
    if not cap_check.isOpened():
        raise IOError(f"Could not open video file for property checking: {video_path}") 
    
    frame_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps_check = cap_check.get(cv2.CAP_PROP_FPS)
    total_frames_approx = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT)) # Can be inaccurate
    
    if frame_width <= 0 or frame_height <= 0: # Fallback if props are zero
        ret_test, test_frame = cap_check.read()
        if ret_test and test_frame is not None:
            frame_height, frame_width = test_frame.shape[:2]
            log.info(f"Read frame dimensions from first frame for VideoWriter: {frame_width}x{frame_height}")
        else:
            cap_check.release()
            raise ValueError("Could not determine video dimensions for VideoWriter.")
    cap_check.release()

    effective_output_fps = fallback_frame_rate
    if output_fps_override and output_fps_override > 0:
        effective_output_fps = output_fps_override
    elif native_fps_check and native_fps_check > 0:
        effective_output_fps = int(round(native_fps_check))
    effective_output_fps = max(1, effective_output_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, float(effective_output_fps), (frame_width, frame_height))
    if not out_writer.isOpened():
        raise IOError(f"Could not create video writer at {output_path}. Check path and permissions.")

    processed_frames_count = 0
    processing_start_time = time.time()
    preview_window_name = "SAHI Tracking Preview (Async)"

    try:
        # stream_video_tracking is now an async iterator
        async for result in stream_tracker.stream_video_tracking(
            video_path=video_path, vid_stride=vid_stride,
            include_annotated_frame=True, show_labels=show_labels,
            line_width=line_width, fallback_frame_rate=fallback_frame_rate
        ):
            if result.annotated_frame is not None:
                out_writer.write(result.annotated_frame)
            else: # Should not happen if include_annotated_frame is True and processing succeeds
                log.warning(f"Frame {result.frame_number} had no annotated_frame to write.")

            processed_frames_count += 1
            
            if show_preview and result.annotated_frame is not None:
                cv2.imshow(preview_window_name, result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): # cv2.waitKey is blocking
                    log.info("Preview quit by user ('q' key pressed).")
                    break
            
            if processed_frames_count % 50 == 0 or processed_frames_count == 1:
                current_elapsed_time = time.time() - processing_start_time
                current_fps_proc = processed_frames_count / current_elapsed_time if current_elapsed_time > 0 else 0
                
                eta_str = "N/A"
                if total_frames_approx > 0 and vid_stride > 0 and current_fps_proc > 0:
                    num_frames_to_process_total = (total_frames_approx / vid_stride)
                    remaining_frames_to_process = num_frames_to_process_total - processed_frames_count
                    if remaining_frames_to_process > 0:
                        eta_seconds = remaining_frames_to_process / current_fps_proc
                        eta_str = f"{eta_seconds:.1f}s"
                
                log.info(f"Frame {result.frame_number}/{total_frames_approx if total_frames_approx > 0 else '?'} | "
                         f"Processed (strided): {processed_frames_count} | "
                         f"FPS: {current_fps_proc:.2f} | ETA: {eta_str}")

    except Exception as e_main:
        log.error(f"An error occurred during ASYNC video processing: {type(e_main).__name__}: {str(e_main)}")
        log.error(f"Full traceback:\n{traceback.format_exc()}")
    finally:
        out_writer.release()
        if show_preview:
            cv2.destroyAllWindows() # Important to clean up

        total_processing_time = time.time() - processing_start_time
        avg_processing_fps = processed_frames_count / total_processing_time if total_processing_time > 0 and processed_frames_count > 0 else 0

        log.info("\n" + "-" * 40 + "\n--- ASYNC Video Processing Finished ---\n" +
                  f"Total frames processed (strided): {processed_frames_count}\n" +
                  f"Total processing time: {total_processing_time:.2f} seconds\n" +
                  f"Average processing FPS: {avg_processing_fps:.2f}\n" +
                  f"Output video saved to: {output_path}\n" + "-" * 40)
    return output_path


async def main():
    # Example usage:
    class_names = {
        0: 'text'
    }


    # --- Configuration ---
    # General paths (adjust as needed)
    try:
        from src import path_to_project # Assuming this exists from your original code
    except ImportError:
        path_to_project = lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        log.warning(f"src.path_to_project not found, using fallback: {path_to_project()}")

    PROJECT_ROOT = Path(path_to_project())
    DOCS_DIR = PROJECT_ROOT / "docs"
    VIDEO_INPUT_PATH = str(DOCS_DIR / "check.mp4") # Make sure this video exists
    # Local model (if used)
    LOCAL_MODEL_PATH = str(DOCS_DIR / "last.engine") # Or .pt, .onnx etc.

    OUTPUT_DIR = DOCS_DIR / "outputs_tracker_integration"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Triton URLs (replace with your actual URLs or set via environment variables)
    # These would typically come from your src.utils.env.Env()
    TRITON_HTTP_URL = os.getenv("TRITON_API_URL", "http://localhost:8000") # Triton HTTP/REST port
    TRITON_WS_URL = os.getenv("TRITON_WS_URL", "ws://localhost:8000")   # Triton WebSocket port for inference service

    # --- Test Scenarios ---
    scenarios = [
        {
            "name": "local_yolo",
            "detection_source": "local",
            "model_path": LOCAL_MODEL_PATH, # Required for local
            "output_filename": "output_local_yolo.mp4",
            "triton_stream_url": None, "triton_ws_url": None, "class_names_map": None
        },
        # {
        #     "name": "triton_stream_yolo",
        #     "detection_source": "triton_stream",
        #     "output_filename": "output_triton_stream.mp4",
        #     "triton_stream_url": TRITON_HTTP_URL, "triton_ws_url": None,
        #     "class_names_map": class_names, # Provide class names for Triton
        #     "model_path": None, # Not used by Triton source
        # },
        # {
        #     "name": "triton_ws_yolo",
        #     "detection_source": "triton_ws",
        #     "output_filename": "output_triton_ws.mp4",
        #     "triton_ws_url": TRITON_WS_URL, "triton_stream_url": None,
        #     "class_names_map": class_names, # Provide class names for Triton
        #     "model_path": None, # Not used by Triton source
        # },
    ]

    if not os.path.exists(VIDEO_INPUT_PATH):
        log.error(f"Video input file not found: {VIDEO_INPUT_PATH}. Skipping main execution.")
        return
    
    for scen in scenarios:
        log.info(f"\n--- Running scenario: {scen['name']} ---")
        
        # Validate paths for current scenario
        if scen['detection_source'] == 'local' and (not scen['model_path'] or not os.path.exists(scen['model_path'])):
            log.warning(f"Local model path for scenario '{scen['name']}' not found: {scen['model_path']}. Skipping.")
            continue
        if scen['detection_source'] == 'triton_stream' and not scen['triton_stream_url']:
            log.warning(f"Triton Stream URL not set for scenario '{scen['name']}'. Skipping.")
            continue
        if scen['detection_source'] == 'triton_ws' and not scen['triton_ws_url']:
            log.warning(f"Triton WebSocket URL not set for scenario '{scen['name']}'. Skipping.")
            continue


        full_output_path = str(OUTPUT_DIR / scen["output_filename"])

        try:
            await track_video_sahi(
                video_path=VIDEO_INPUT_PATH,
                output_path=full_output_path,
                model_path=scen.get("model_path"), # Will be None if not in scen dict
                tracker_type='botsort',
                fallback_frame_rate=30,
                window_size_ratio=(0.7, 0.7), overlap_ratio=(0.1, 0.1),
                img_size=640, conf=0.1, iou=0.1, nms_global=0.1, # Adjust thresholds as needed
                show_labels=True, line_width=2,
                show_preview=False, # Set to True for GUI preview (beware of asyncio/cv2 issues)
                vid_stride=1, 
                classes=[0], # Filter classes for local model if needed, e.g., [0] for persons
                detection_source=scen["detection_source"],
                triton_stream_url=scen.get("triton_stream_url"),
                triton_ws_url=scen.get("triton_ws_url"),
                triton_model_name="yolo", # Your Triton model name for YOLO
                class_names_map=scen.get("class_names_map")
            )
            log.info(f"Scenario '{scen['name']}' completed. Output at: {full_output_path}")
        except Exception as e_scen:
            log.error(f"Error in scenario '{scen['name']}': {type(e_scen).__name__}: {str(e_scen)}")
            log.error(f"Full traceback for scenario error:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # Setup basic logging for the example
    import logging
    logging.basicConfig(level=logging.INFO) # General logging
    # log.setLevel(logging.DEBUG) # Set our specific logger to DEBUG for more details
    
    asyncio.run(main())
