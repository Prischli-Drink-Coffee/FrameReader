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

from src.triton_api.main_endpoint import MainEndpointClient 
from src.triton_api.stream_endpoint import StreamEndpointClient
from src.triton_api.websocket_endpoint import WebSocketEndpointClient


from src.utils.custom_logging import setup_logging

log = setup_logging()

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

        self.data = torch.from_numpy(boxes_data_np).float().cpu()
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

    def cpu(self):
        return self

    def __len__(self):
        return self.data.shape[0]


class MockResults:
    def __init__(self, boxes_data_np: np.ndarray, orig_shape: Tuple[int, int]):
        self.boxes = MockBoxesData(boxes_data_np, orig_shape)
        self.conf = self.boxes.conf
        self.xywh = self.boxes.xywh
        self.cls = self.boxes.cls
        self.names = {}
        self.masks = None
        self.probs = None
        self.keypoints = None
        self.orig_shape = orig_shape
        self.orig_img = None

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
        iou_threshold: float = 0.45,
        nms_threshold_global: float = 0.5,
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
        detection_source: str = "local",
        triton_stream_url: Optional[str] = None,
        triton_ws_url: Optional[str] = None,
        triton_batch_url: Optional[str] = None,
        triton_model_name: str = "yolo",
        triton_chunk_size: int = 1,
        triton_max_batch_size: int = 16,
        class_names_map: Optional[Dict[int, str]] = None
    ):
        self.detection_source = detection_source
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.triton_max_batch_size = triton_max_batch_size
        
        log.debug(f"Using device: {self.device}, max_batch_size: {triton_max_batch_size}")

        self.model_names: Dict[int, str] = {}
        self.model_names_reverse_map: Dict[str, int] = {}
        
        self.triton_stream_client: Optional[StreamEndpointClient] = None
        self.triton_ws_client: Optional[WebSocketEndpointClient] = None
        self.triton_batch_client: Optional[MainEndpointClient] = None

        self._initialize_detection_source(
            model_path, triton_stream_url, triton_ws_url, triton_batch_url,
            triton_chunk_size, class_names_map
        )
        
        self.triton_model_name = triton_model_name
        self.tracker_type = tracker_type

        self._initialize_tracker(tracker_type, tracker_config_path, frame_rate)
        
        self.window_size_ratio = window_size_ratio
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold_global = nms_threshold_global
        self.classes = classes

    def _initialize_detection_source(
        self, 
        model_path: Optional[str], 
        triton_stream_url: Optional[str], 
        triton_ws_url: Optional[str], 
        triton_batch_url: Optional[str],
        triton_chunk_size: int,
        class_names_map: Optional[Dict[int, str]]
    ) -> None:
        if self.detection_source == "local":
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            log.debug(f"Loading local model from {model_path}")
            self.model = YOLO(model_path, task='detect')
            self.model_names = self.model.names
            
        elif self.detection_source == "stream":
            if not triton_stream_url:
                raise ValueError("Triton stream URL required for stream source")
            self.triton_stream_client = StreamEndpointClient(
                base_url=triton_stream_url, 
                chunk_size=triton_chunk_size
            )
            log.debug(f"Triton Stream client: {triton_stream_url}")
            
        elif self.detection_source == "ws":
            if not triton_ws_url:
                raise ValueError("Triton WebSocket URL required for ws source")
            self.triton_ws_client = WebSocketEndpointClient(base_url=triton_ws_url)
            log.debug(f"Triton WebSocket client: {triton_ws_url}")
            
        elif self.detection_source == "main":
            if not triton_batch_url:
                raise ValueError("Triton batch URL required for main source")
            self.triton_batch_client = MainEndpointClient(base_url=triton_batch_url)
            log.debug(f"Triton Batch client: {triton_batch_url}")
            
        else:
            raise ValueError(f"Unsupported detection_source: {self.detection_source}")

        if class_names_map and self.detection_source != "local":
            self.model_names = class_names_map
            self.model_names_reverse_map = {v: k for k, v in class_names_map.items()}
        elif not self.model_names and self.detection_source != "local":
            log.warning("class_names_map not provided for Triton source")

    def _initialize_tracker(
        self, 
        tracker_type: str, 
        tracker_config_path: Optional[str], 
        frame_rate: int
    ) -> None:
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
                    f"Tracker config not found: '{tracker_type}.yaml'"
                ) from e
        
        tracker_cfg_dict = YAML.load(str(tracker_config_file))
        tracker_cfg_dict['tracker_type'] = tracker_type
        
        tracker_cfg = IterableSimpleNamespace(**tracker_cfg_dict)
        self.tracker_cfg_dict = tracker_cfg_dict
        self.current_frame_rate = frame_rate

        if tracker_cfg.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError(f"Unsupported tracker type: {tracker_cfg.tracker_type}")

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](
            args=tracker_cfg, frame_rate=frame_rate
        )
        log.info(f"Initialized {tracker_cfg.tracker_type} tracker (fps={frame_rate})")

    def _get_class_id(self, class_name: str) -> int:
        if not self.model_names_reverse_map:
            import hashlib
            return int(hashlib.md5(class_name.encode()).hexdigest(), 16) % 1000 
        
        class_id = self.model_names_reverse_map.get(class_name)
        if class_id is None:
            import hashlib
            return int(hashlib.md5(class_name.encode()).hexdigest(), 16) % 1000
        return class_id

    def _prepare_batch_windows(
        self, 
        frame: np.ndarray, 
        target_size_w: int, 
        target_size_h: int
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:

        frame_height, frame_width = frame.shape[:2]
        windows = self._get_windows(frame_width, frame_height)
        
        batch_images = []
        window_coords_list = []
        
        for x1_w, y1_w, x2_w, y2_w in windows:
            window_slice = frame[y1_w:y2_w, x1_w:x2_w]
            if window_slice.size == 0:
                continue
            
            window_coords_list.append((x1_w, y1_w, x2_w, y2_w))
            
            if (target_size_w != window_slice.shape[1] or 
                target_size_h != window_slice.shape[0]):
                resized_window = cv2.resize(window_slice, (target_size_w, target_size_h))
            else:
                resized_window = window_slice.copy()
                
            batch_images.append(resized_window)
        
        return batch_images, window_coords_list

    async def _process_triton_batch(
        self, 
        batch_images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        
        if self.detection_source == "triton_stream" and self.triton_stream_client:
            async with self.triton_stream_client as client:
                result = await client.stream_collect_from_arrays(
                    image_arrays=batch_images,
                    model_name=self.triton_model_name
                )
                return result.get('results', [])
                
        elif self.detection_source == "triton_ws" and self.triton_ws_client:
            return await self.triton_ws_client.run_inference_session(
                model_name=self.triton_model_name,
                images=batch_images
            )
            
        elif self.detection_source == "triton_batch" and self.triton_batch_client:
            async with self.triton_batch_client as client:
                if self.triton_model_name == "yolo":
                    result = await client.yolo_inference_from_arrays(
                        image_arrays=batch_images,
                        filenames=[f"window_{i}.jpg" for i in range(len(batch_images))]
                    )
                else:
                    result = await client.donut_inference_from_arrays(
                        image_arrays=batch_images,
                        filenames=[f"window_{i}.jpg" for i in range(len(batch_images))]
                    )
                return result.get('results', [])
        
        return []

    async def _process_chunked_batch(
        self, 
        batch_images: List[np.ndarray], 
        chunk_size: int
    ) -> List[Dict[str, Any]]:

        all_results = []
        
        for i in range(0, len(batch_images), chunk_size):
            chunk = batch_images[i:i + chunk_size]
            chunk_results = await self._process_triton_batch(chunk)
            all_results.extend(chunk_results)
            
        return all_results

    def _adapt_detection_format(
        self,
        triton_results: List[Dict[str, Any]],
        window_coords_list: List[Tuple[int, int, int, int]]
    ) -> List[Dict[str, torch.Tensor]]:
        adapted_results: List[Dict[str, torch.Tensor]] = []
        for i, result_data in enumerate(triton_results):
            if i >= len(window_coords_list):
                break
            detections = result_data.get('detections')
            if detections is None:
                boxes = result_data.get('boxes', [])
                confidences = result_data.get('confidences', [])
                classes = result_data.get('classes', [])
                detections = []
                for box, score, cls in zip(boxes, confidences, classes):
                    label = self.model_names.get(int(cls), str(cls))
                    detections.append({
                        'box2d': box,
                        'score': score,
                        'label': label
                    })
            win_x1, win_y1, win_x2, win_y2 = window_coords_list[i]
            win_w, win_h = win_x2 - win_x1, win_y2 - win_y1
            adapted = self._adapt_triton_detections_to_torch(detections, win_w, win_h)
            adapted_results.append(adapted)
        while len(adapted_results) < len(window_coords_list):
            adapted_results.append(self._adapt_triton_detections_to_torch([], 0, 0))
        return adapted_results

    def _adapt_triton_detections_to_torch(
        self,
        triton_detections: List[Dict[str, Any]],
        window_orig_w: int,
        window_orig_h: int
    ) -> Dict[str, torch.Tensor]:
        if not triton_detections:
            return {
                'xyxy': torch.empty((0, 4), device=self.device, dtype=torch.float32),
                'conf': torch.empty(0, device=self.device, dtype=torch.float32),
                'cls': torch.empty((0,), device=self.device, dtype=torch.int32)
            }
        xyxys, confs, clss = [], [], []
        for det in triton_detections:
            box = det['box2d']
            xyxys.append(box)
            confs.append(det['score'])
            clss.append(self._get_class_id(det['label']))
        return {
            'xyxy': torch.tensor(xyxys, device=self.device, dtype=torch.float32),
            'conf': torch.tensor(confs, device=self.device, dtype=torch.float32),
            'cls': torch.tensor(clss, device=self.device, dtype=torch.int32)
        }

    def _scale_detections_to_frame(
        self,
        per_window_outputs: List[Dict[str, torch.Tensor]],
        window_coords_list: List[Tuple[int, int, int, int]], 
        target_size_w: int,
        target_size_h: int,
        frame_width: int,
        frame_height: int
    ) -> List[Dict[str, Any]]:

        all_detections = []
        
        for i, model_output in enumerate(per_window_outputs):
            if i >= len(window_coords_list):
                continue
                
            boxes_xyxy = model_output['xyxy']
            confs = model_output['conf'] 
            clss = model_output['cls']

            if boxes_xyxy.numel() == 0:
                continue

            win_x1, win_y1, win_x2, win_y2 = window_coords_list[i]
            orig_win_w, orig_win_h = win_x2 - win_x1, win_y2 - win_y1

            for j in range(boxes_xyxy.shape[0]):
                box_resized = boxes_xyxy[j].tolist()
                conf = confs[j].item()
                cls_id = clss[j].item()

                f_x1 = (box_resized[0] / target_size_w) * orig_win_w + win_x1
                f_y1 = (box_resized[1] / target_size_h) * orig_win_h + win_y1
                f_x2 = (box_resized[2] / target_size_w) * orig_win_w + win_x1
                f_y2 = (box_resized[3] / target_size_h) * orig_win_h + win_y1
                
                f_x1 = max(0.0, min(f_x1, float(frame_width)))
                f_y1 = max(0.0, min(f_y1, float(frame_height)))
                f_x2 = max(0.0, min(f_x2, float(frame_width)))
                f_y2 = max(0.0, min(f_y2, float(frame_height)))
                
                if f_x2 > f_x1 and f_y2 > f_y1:
                    all_detections.append({
                        "bbox": [f_x1, f_y1, f_x2 - f_x1, f_y2 - f_y1],
                        "confidence": conf,
                        "class_id": cls_id,
                        "xyxy": [f_x1, f_y1, f_x2, f_y2]
                    })
                    
        return all_detections

    async def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:

        frame_height, frame_width = frame.shape[:2]
        
        if isinstance(self.img_size, int):
            target_size_w = target_size_h = self.img_size
        else:
            target_size_w, target_size_h = self.img_size

        batch_images, window_coords_list = self._prepare_batch_windows(
            frame, target_size_w, target_size_h
        )
        
        if not batch_images:
            return []

        per_window_outputs: List[Dict[str, torch.Tensor]] = []

        try:
            if self.detection_source == "local":
                results = self.model.predict(
                    source=batch_images,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=(target_size_h, target_size_w),
                    classes=self.classes,
                    device=self.device,
                    verbose=False,
                    augment=False
                )
                
                for res_obj in results:
                    per_window_outputs.append({
                        'xyxy': res_obj.boxes.xyxy.to(self.device),
                        'conf': res_obj.boxes.conf.to(self.device),
                        'cls': res_obj.boxes.cls.to(self.device).int()
                    })
                    
            else:
                if len(batch_images) <= self.triton_max_batch_size:
                    triton_results = await self._process_triton_batch(batch_images)
                else:
                    triton_results = await self._process_chunked_batch(
                        batch_images, self.triton_max_batch_size
                    )
                
                per_window_outputs = self._adapt_detection_format(
                    triton_results, window_coords_list
                )

            all_detections = self._scale_detections_to_frame(
                per_window_outputs, window_coords_list,
                target_size_w, target_size_h, frame_width, frame_height
            )

        except Exception as e:
            log.error(f"Detection processing error ({self.detection_source}): {e}")
            return []

        merged_detections_np = self._nms_global(all_detections)
        mock_results = MockResults(merged_detections_np, (frame_height, frame_width))
        mock_results.names = self.model_names

        try:
            tracked_output_np = self.tracker.update(mock_results, frame)
        except Exception:
            return []

        return self._format_tracking_results(
            tracked_output_np, frame_width, frame_height
        )

    def _format_tracking_results(
        self, 
        tracked_output_np: np.ndarray, 
        frame_width: int, 
        frame_height: int
    ) -> List[Dict[str, Any]]:
        tracked_objects = []
        
        if not isinstance(tracked_output_np, np.ndarray) or tracked_output_np.size == 0:
            return tracked_objects
            
        output_cols = tracked_output_np.shape[1]
        if output_cols < 7:
            return tracked_objects
            
        for row in tracked_output_np:
            x1, y1, x2, y2, track_id, conf, cls_id = row[:7]
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id, cls_id = int(track_id), int(cls_id)

            x1 = max(0, min(x1, frame_width))
            y1 = max(0, min(y1, frame_height))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = self.model_names.get(cls_id, f"Cls_{cls_id}")
            obj_data = {
                'box': [x1, y1, x2, y2],
                'track_id': track_id,
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': float(conf)
            }
            
            if output_cols >= 8 and self.tracker_type == 'botsort':
                tracker_cfg = self.tracker.args.get('tracker_cfg', {})
                if 'idx' in tracker_cfg.get('public_vars', []):
                    obj_data['original_det_idx'] = int(row[7])
            
            tracked_objects.append(obj_data)

        return tracked_objects

    def _get_windows(self, frame_width: int, frame_height: int) -> List[Tuple[int, int, int, int]]:
        win_w = max(1, int(frame_width * self.window_size_ratio[0]))
        win_h = max(1, int(frame_height * self.window_size_ratio[1]))
        
        overlap_w = int(win_w * self.overlap_ratio[0])
        overlap_h = int(win_h * self.overlap_ratio[1])

        step_w = max(1, win_w - overlap_w)
        step_h = max(1, win_h - overlap_h)

        windows = []
        for y_start in range(0, frame_height, step_h):
            y1 = min(y_start, frame_height - win_h) if y_start + win_h > frame_height else y_start
            y2 = min(y1 + win_h, frame_height)
            
            if y1 >= y2:
                continue

            for x_start in range(0, frame_width, step_w):
                x1 = min(x_start, frame_width - win_w) if x_start + win_w > frame_width else x_start
                x2 = min(x1 + win_w, frame_width)
                
                if x1 >= x2:
                    continue

                window_coords = (x1, y1, x2, y2)
                if window_coords not in windows:
                    windows.append(window_coords)

                if x2 == frame_width:
                    break
            if y2 == frame_height:
                break
        
        if not windows and frame_width > 0 and frame_height > 0:
            windows.append((0, 0, frame_width, frame_height))

        return windows

    def _nms_global(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        detection_tensor = torch.tensor([
            [det['xyxy'][0], det['xyxy'][1], det['xyxy'][2], det['xyxy'][3], 
             det['confidence'], det['class_id']]
            for det in detections
        ], dtype=torch.float32, device=self.device)

        if detection_tensor.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = detection_tensor[:, :4]
        scores = detection_tensor[:, 4]
        classes = detection_tensor[:, 5]

        final_detections = []
        
        for cls_id in torch.unique(classes):
            cls_mask = (classes == cls_id)
            if not torch.any(cls_mask):
                continue

            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            keep_indices = torchvision.ops.nms(
                cls_boxes, cls_scores, self.nms_threshold_global
            )
            
            original_indices = torch.where(cls_mask)[0]
            final_detections.append(detection_tensor[original_indices[keep_indices]])

        if not final_detections:
            return np.empty((0, 6), dtype=np.float32)

        return torch.cat(final_detections, dim=0).cpu().numpy()

    def annotate_frame(
        self, 
        frame: np.ndarray, 
        tracked_objects: List[Dict[str, Any]], 
        show_labels: bool = True, 
        line_width: int = 2
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for obj in tracked_objects:
            box = obj['box']
            track_id = obj['track_id']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            x1, y1, x2, y2 = map(int, box)
            color = get_color(track_id)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_width)
            
            if show_labels:
                label_text = f"ID:{track_id} {class_name} ({confidence:.2f})"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                
                label_bg_y1 = max(y1 - text_h - baseline - 3, 0)
                label_bg_y2 = y1 - 3
                text_y_pos = max(y1 - baseline - 3, text_h)

                cv2.rectangle(
                    annotated_frame, (x1, label_bg_y1), (x1 + text_w, label_bg_y2), 
                    color, -1
                )
                cv2.putText(
                    annotated_frame, label_text, (x1, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )
                
        return annotated_frame

    def reset_tracker(self) -> None:
        log.info("Resetting tracker state")
        
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        else:
            log.warning("Tracker has no reset method. Re-initializing...")
            tracker_cfg = IterableSimpleNamespace(**self.tracker_cfg_dict)
            TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
            self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](
                args=tracker_cfg, frame_rate=self.current_frame_rate
            )


class VideoStreamTracker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        tracker_type: str = 'botsort',
        tracker_config_path: Optional[str] = None,
        window_size_ratio: Tuple[float, float] = (0.7, 0.7),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        img_size: Union[int, Tuple[int,int]] = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        nms_global: float = 0.5,
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None,
        detection_source: str = "local",
        triton_stream_url: Optional[str] = None,
        triton_ws_url: Optional[str] = None,
        triton_batch_url: Optional[str] = None,
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
        self.triton_batch_url = triton_batch_url
        self.triton_model_name = triton_model_name
        self.triton_chunk_size = triton_chunk_size
        self.class_names_map = class_names_map
        
        self.tracker_wrapper: Optional[SAHITrackingWrapper] = None
        self.current_frame_rate_stored = 30

    def _initialize_tracker(self, fps: int):
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
                triton_batch_url=self.triton_batch_url,
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
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    raise ValueError("Could not read valid frame dimensions from video.")
            else:
                frame_width, frame_height = frame_width_prop, frame_height_prop

            effective_fps = int(round(native_fps)) if native_fps and native_fps > 0 else fallback_frame_rate
            effective_fps = max(1, effective_fps)

            self._initialize_tracker(effective_fps)

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
                
                if self.tracker_wrapper is None:
                    raise RuntimeError("Tracker wrapper not initialized.")
                
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
        camera_index: Union[int, str] = 0,
        include_annotated_frame: bool = False,
        show_labels: bool = True,
        line_width: int = 2,
        frame_rate: int = 30
    ) -> AsyncIterator[FrameTrackingResult]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Could not open camera with index/path: {camera_index}")

        cam_frame_height, cam_frame_width = -1,-1
        try:

            cap.set(cv2.CAP_PROP_FPS, float(frame_rate)) 
            
            ret_test, test_frame = cap.read()
            if not ret_test or test_frame is None:
                raise IOError(f"Could not read initial frame from camera {camera_index}")
            cam_frame_height, cam_frame_width = test_frame.shape[:2]
            
            self._initialize_tracker(frame_rate)

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

    def reset_tracker(self):
        if self.tracker_wrapper:
            self.tracker_wrapper.reset_tracker()


async def track_video_sahi(
    video_path: str,
    output_path: str,
    model_path: Optional[str] = None,
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
    detection_source: str = "local",
    triton_stream_url: Optional[str] = None,
    triton_ws_url: Optional[str] = None,
    triton_batch_url: Optional[str] = None,
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
        triton_ws_url=triton_ws_url, triton_batch_url=triton_batch_url,
        triton_model_name=triton_model_name, triton_chunk_size=triton_chunk_size, 
        class_names_map=class_names_map
    )

    cap_check = cv2.VideoCapture(video_path)
    if not cap_check.isOpened():
        raise IOError(f"Could not open video file for property checking: {video_path}") 
    
    frame_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps_check = cap_check.get(cv2.CAP_PROP_FPS)
    total_frames_approx = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_width <= 0 or frame_height <= 0:
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
        async for result in stream_tracker.stream_video_tracking(
            video_path=video_path, vid_stride=vid_stride,
            include_annotated_frame=True, show_labels=show_labels,
            line_width=line_width, fallback_frame_rate=fallback_frame_rate
        ):
            if result.annotated_frame is not None:
                out_writer.write(result.annotated_frame)
            else:
                log.warning(f"Frame {result.frame_number} had no annotated_frame to write.")

            processed_frames_count += 1
            
            if show_preview and result.annotated_frame is not None:
                cv2.imshow(preview_window_name, result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
            cv2.destroyAllWindows()

        total_processing_time = time.time() - processing_start_time
        avg_processing_fps = processed_frames_count / total_processing_time if total_processing_time > 0 and processed_frames_count > 0 else 0

        log.info("\n" + "-" * 40 + "\n--- ASYNC Video Processing Finished ---\n" +
                  f"Total frames processed (strided): {processed_frames_count}\n" +
                  f"Total processing time: {total_processing_time:.2f} seconds\n" +
                  f"Average processing FPS: {avg_processing_fps:.2f}\n" +
                  f"Output video saved to: {output_path}\n" + "-" * 40)
    return output_path


async def main():

    class_names = {
        0: 'text'
    }

    try:
        from src import path_to_project
    except ImportError:
        path_to_project = lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        log.warning(f"src.path_to_project not found, using fallback: {path_to_project()}")

    PROJECT_ROOT = Path(path_to_project())
    DOCS_DIR = PROJECT_ROOT / "docs"
    VIDEO_INPUT_PATH = str(DOCS_DIR / "check.mp4")
    LOCAL_MODEL_PATH = str(DOCS_DIR / "last.engine")

    OUTPUT_DIR = DOCS_DIR / "outputs_tracker_integration"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TRITON_HTTP_URL = os.getenv("TRITON_API_URL", "http://localhost:8000")
    TRITON_WS_URL = os.getenv("TRITON_WS_URL", "ws://localhost:8000")

    scenarios = [
        # {
        #     "name": "local_yolo",
        #     "detection_source": "local",
        #     "model_path": LOCAL_MODEL_PATH, # Required for local
        #     "output_filename": "output_local_yolo.mp4",
        #     "triton_stream_url": None, "triton_ws_url": None, "triton_batch_url": None,
        #     "class_names_map": None
        # },
        {
            "name": "triton_batch_yolo",
            "detection_source": "main",
            "model_path": None, # Not used by Triton source
            "output_filename": "output_triton_batch.mp4",
            "triton_stream_url": None, "triton_ws_url": None,
            "triton_batch_url": TRITON_HTTP_URL, # Main endpoint URL
            "class_names_map": class_names # Provide class names for Triton
        },
        # {
        #     "name": "triton_stream_yolo",
        #     "detection_source": "stream",
        #     "output_filename": "output_triton_stream.mp4",
        #     "triton_stream_url": TRITON_HTTP_URL, "triton_ws_url": None, "triton_batch_url": None,
        #     "class_names_map": class_names, # Provide class names for Triton
        #     "model_path": None, # Not used by Triton source
        # },
        # {
        #     "name": "triton_ws_yolo",
        #     "detection_source": "ws",
        #     "output_filename": "output_triton_ws.mp4",
        #     "triton_ws_url": TRITON_WS_URL, "triton_stream_url": None, "triton_batch_url": None,
        #     "class_names_map": class_names, # Provide class names for Triton
        #     "model_path": None, # Not used by Triton source
        # },
    ]

    if not os.path.exists(VIDEO_INPUT_PATH):
        log.error(f"Video input file not found: {VIDEO_INPUT_PATH}. Skipping main execution.")
        return
    
    for scen in scenarios:
        log.info(f"\n--- Running scenario: {scen['name']} ---")
        
        if scen['detection_source'] == 'local' and (not scen['model_path'] or not os.path.exists(scen['model_path'])):
            log.warning(f"Local model path for scenario '{scen['name']}' not found: {scen['model_path']}. Skipping.")
            continue
        if scen['detection_source'] == 'stream' and not scen['triton_stream_url']:
            log.warning(f"Triton Stream URL not set for scenario '{scen['name']}'. Skipping.")
            continue
        if scen['detection_source'] == 'ws' and not scen['triton_ws_url']:
            log.warning(f"Triton WebSocket URL not set for scenario '{scen['name']}'. Skipping.")
            continue
        if scen['detection_source'] == 'main' and not scen['triton_batch_url']:
            log.warning(f"Triton Batch URL not set for scenario '{scen['name']}'. Skipping.")
            continue

        full_output_path = str(OUTPUT_DIR / scen["output_filename"])

        try:
            await track_video_sahi(
                video_path=VIDEO_INPUT_PATH,
                output_path=full_output_path,
                model_path=scen.get("model_path"),
                tracker_type='botsort',
                fallback_frame_rate=30,
                window_size_ratio=(0.7, 0.7), overlap_ratio=(0.1, 0.1),
                img_size=640, conf=0.1, iou=0.1, nms_global=0.1,
                show_labels=True, line_width=2,
                show_preview=False,
                vid_stride=1, 
                classes=[0],
                detection_source=scen["detection_source"],
                triton_stream_url=scen.get("triton_stream_url"),
                triton_ws_url=scen.get("triton_ws_url"),
                triton_batch_url=scen.get("triton_batch_url"),
                triton_model_name="yolo",
                class_names_map=scen.get("class_names_map")
            )
            log.info(f"Scenario '{scen['name']}' completed. Output at: {full_output_path}")
        except Exception as e_scen:
            log.error(f"Error in scenario '{scen['name']}': {type(e_scen).__name__}: {str(e_scen)}")
            log.error(f"Full traceback for scenario error:\n{traceback.format_exc()}")


if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())
