import os
import cv2
import numpy as np
import torch
import torchvision
import time
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator, NamedTuple
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker, BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from pathlib import Path
from dataclasses import dataclass
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


class MockBoxesData:
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
        model_path: str,
        tracker_type: str = 'botsort',
        tracker_config_path: Optional[str] = None,
        frame_rate: int = 30,
        window_size_ratio: Tuple[float, float] = (0.7, 0.7),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_threshold_global: float = 0.5,
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.debug(f"Using device: {self.device}")

        log.debug(f"Loading detection model from {model_path}...")
        self.model = YOLO(model_path, task='detect')
        self.model_names = self.model.names
        log.debug("Detection model loaded.")

        script_dir = Path(__file__).parent.resolve()
        default_cfg_path = script_dir / 'cfg' / 'trackers' / f'{tracker_type}.yaml'

        if tracker_config_path and os.path.exists(tracker_config_path):
            tracker_config_file = tracker_config_path
            log.debug(f"Using custom tracker config: {tracker_config_file}")
        elif default_cfg_path.exists():
            tracker_config_file = str(default_cfg_path)
            log.debug(f"Using default tracker config relative to script: {tracker_config_file}")
        else:
            try:
                tracker_config_file = check_yaml(f'{tracker_type}.yaml')
                log.debug(f"Using ultralytics default tracker config found via check_yaml: {tracker_config_file}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not find tracker config: '{tracker_type}.yaml'. "
                                        f"Checked relative path, CWD, and ultralytics defaults. "
                                        f"Provide a valid 'tracker_config_path'.") from e

        tracker_cfg_dict = YAML.load(str(tracker_config_file))
        if tracker_cfg_dict.get('tracker_type') != tracker_type:
            log.warning(f"Warning: Tracker config file specifies type '{tracker_cfg_dict.get('tracker_type')}', "
                  f"but '{tracker_type}' was requested. Overriding config type to '{tracker_type}'.")
            tracker_cfg_dict['tracker_type'] = tracker_type
        tracker_cfg = IterableSimpleNamespace(**tracker_cfg_dict)

        if tracker_cfg.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError(f"Unsupported tracker type: {tracker_cfg.tracker_type}")

        if tracker_type == 'botsort' and getattr(tracker_cfg, 'with_reid', False):
            log.warning("Warning: BoT-SORT with ReID features are not explicitly handled by this SAHI wrapper.")

        log.info(f"Initializing {tracker_cfg.tracker_type} tracker (frame_rate={frame_rate})...")
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](args=tracker_cfg, frame_rate=frame_rate)

        self.window_size_ratio = window_size_ratio
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold_global = nms_threshold_global
        self.classes = classes

    def _get_windows(self, frame_width: int, frame_height: int) -> List[Tuple[int, int, int, int]]:
        win_w = max(1, int(frame_width * self.window_size_ratio[0]))
        win_h = max(1, int(frame_height * self.window_size_ratio[1]))
        overlap_w = int(win_w * self.overlap_ratio[0])
        overlap_h = int(win_h * self.overlap_ratio[1])

        step_w = max(1, win_w - overlap_w)
        step_h = max(1, win_h - overlap_h)

        windows = []
        for y1 in range(0, frame_height, step_h):
            y2 = min(y1 + win_h, frame_height)
            current_y1 = y1 if (y1 + win_h <= frame_height) else max(0, frame_height - win_h)
            current_y2 = current_y1 + win_h

            for x1 in range(0, frame_width, step_w):
                x2 = min(x1 + win_w, frame_width)
                current_x1 = x1 if (x1 + win_w <= frame_width) else max(0, frame_width - win_w)
                current_x2 = current_x1 + win_w

                window_coords = (current_x1, current_y1, current_x2, current_y2)
                if window_coords not in windows and current_x2 > current_x1 and current_y2 > current_y1:
                    windows.append(window_coords)

                if x2 == frame_width: break
            if y2 == frame_height: break

        return windows

    def _nms_global(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        detection_list = []
        for det in detections:
            xyxy = det['xyxy']
            detection_list.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], det['confidence'], det['class_id']])

        device_for_nms = self.device if torch.cuda.is_available() else torch.device('cpu')
        detections_tensor = torch.tensor(detection_list, dtype=torch.float32, device=device_for_nms)

        if detections_tensor.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = detections_tensor[:, :4]
        scores = detections_tensor[:, 4]
        classes = detections_tensor[:, 5]

        final_detections_list = []
        unique_classes = torch.unique(classes)

        for cls_id in unique_classes:
            cls_mask = (classes == cls_id)
            if not torch.any(cls_mask): continue

            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_original_indices = torch.where(cls_mask)[0]

            keep_indices_cls = torchvision.ops.nms(cls_boxes, cls_scores, self.nms_threshold_global)

            original_indices_kept = cls_original_indices[keep_indices_cls]
            final_detections_list.append(detections_tensor[original_indices_kept])

        if not final_detections_list:
            return np.empty((0, 6), dtype=np.float32)

        merged_detections_tensor = torch.cat(final_detections_list, dim=0)
        return merged_detections_tensor.cpu().numpy()

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        frame_height, frame_width = frame.shape[:2]
        windows = self._get_windows(frame_width, frame_height)
        all_detections_in_frame_coords = []
        
        if not windows:
            return []
            
        batch_size = 16
        batch_images = []
        window_coords = []
        
        for x1, y1, x2, y2 in windows:
            window_img = frame[y1:y2, x1:x2]
            if window_img.size < 4:
                continue
                
            window_coords.append((x1, y1, x2, y2))
            
            if self.img_size != (window_img.shape[1], window_img.shape[0]):
                window_img = cv2.resize(window_img, (self.img_size, self.img_size))
                
            batch_images.append(window_img)

        num_real_windows = len(batch_images)
        
        if num_real_windows == 0:
            return []
        
        if num_real_windows < batch_size:
            zero_shape = batch_images[0].shape
            zero_img = np.zeros(zero_shape, dtype=np.uint8)
            
            batch_images.extend([zero_img.copy() for _ in range(batch_size - num_real_windows)])
        
        try:
            batch_input = np.array(batch_images)

            batch_results = self.model.predict(
                source=batch_input,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                classes=self.classes,
                device=self.device,
                verbose=False,
                augment=False
            )
            
            for idx in range(num_real_windows):
                if batch_results[idx] and hasattr(batch_results[idx], 'boxes') and batch_results[idx].boxes is not None:
                    boxes_obj = batch_results[idx].boxes
                    x1, y1, x2, y2 = window_coords[idx]
                    
                    for j in range(len(boxes_obj)):
                        w_box = boxes_obj.xyxy[j].tolist()
                        conf = boxes_obj.conf[j].item()
                        cls_id = int(boxes_obj.cls[j].item())
                        
                        f_x1, f_y1, f_x2, f_y2 = w_box[0] + x1, w_box[1] + y1, w_box[2] + x1, w_box[3] + y1
                        f_x1, f_y1 = max(0, f_x1), max(0, f_y1)
                        f_x2, f_y2 = min(frame_width, f_x2), min(frame_height, f_y2)
                        
                        if f_x2 > f_x1 and f_y2 > f_y1:
                            detection = {
                                "bbox": [f_x1, f_y1, f_x2 - f_x1, f_y2 - f_y1],
                                "confidence": conf,
                                "class_id": cls_id,
                                "xyxy": [f_x1, f_y1, f_x2, f_y2]
                            }
                            all_detections_in_frame_coords.append(detection)
        
        except Exception as e:
            log.warning(f"Warning: Error processing batch: {type(e).__name__}: {str(e)}")

        merged_detections_np = self._nms_global(all_detections_in_frame_coords)

        mock_results_for_tracker = MockResults(merged_detections_np, (frame_height, frame_width))

        try:
            tracked_output_np = self.tracker.update(mock_results_for_tracker, frame)
        except Exception as e:
            import traceback
            return []

        tracked_objects_list = []
        if isinstance(tracked_output_np, np.ndarray) and tracked_output_np.size > 0:
            output_cols = tracked_output_np.shape[1]

            if output_cols == 8:
                for row in tracked_output_np:
                    x1, y1, x2, y2, track_id, conf, cls_id, idx = row
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id, cls_id = int(track_id), int(cls_id)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)

                    if x2 > x1 and y2 > y1:
                        class_name = self.model_names.get(cls_id, f"Cls_{cls_id}")
                        tracked_objects_list.append({
                            'box': [x1, y1, x2, y2],
                            'track_id': track_id,
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'original_det_idx': int(idx)
                        })

            elif output_cols == 7:
                for row in tracked_output_np:
                    x1, y1, x2, y2, track_id, conf, cls_id = row
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id, cls_id = int(track_id), int(cls_id)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)
                    if x2 > x1 and y2 > y1:
                        class_name = self.model_names.get(cls_id, f"Cls_{cls_id}")
                        tracked_objects_list.append({
                            'box': [x1, y1, x2, y2],
                            'track_id': track_id,
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': float(conf)
                        })

            elif output_cols == 9:
                for row in tracked_output_np:
                    cx, cy, w, h, angle, track_id, conf, cls_id, idx = row
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                    track_id, cls_id = int(track_id), int(cls_id)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)

                    if x2 > x1 and y2 > y1:
                        class_name = self.model_names.get(cls_id, f"Cls_{cls_id}")
                        tracked_objects_list.append({
                            'box': [x1, y1, x2, y2],
                            'track_id': track_id,
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'original_det_idx': int(idx),
                            'angle': float(angle)
                        })

            else:
                log.warning(f"Warning: Tracker output NumPy array has unexpected shape {tracked_output_np.shape}. Expected 7, 8, or 9 columns.")

        elif isinstance(tracked_output_np, (np.ndarray, list)) and len(tracked_output_np) == 0:
            pass
        else:
            log.warning(f"Warning: Tracker returned unexpected output type: {type(tracked_output_np)}")

        return tracked_objects_list

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
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_y = max(y1 - 10, text_h + baseline + 3)
                label_x = max(x1, 0)
                cv2.rectangle(annotated_frame, (label_x, label_y - text_h - baseline),
                              (label_x + text_w, label_y), color, -1)
                cv2.putText(annotated_frame, label_text, (label_x, label_y - (baseline//2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame

    def reset_tracker(self):
        log.info("Resetting tracker state...")
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        else:
            log.warning("Warning: Tracker object does not have a .reset() method.")


class VideoStreamTracker:
    def __init__(
        self,
        model_path: str,
        tracker_type: str = 'botsort',
        tracker_config_path: Optional[str] = None,
        window_size_ratio: Tuple[float, float] = (0.7, 0.7),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        img_size: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        nms_global: float = 0.5,
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None
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
        self.tracker_wrapper = None

    def _initialize_tracker(self, fps: int):
        if self.tracker_wrapper is None:
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
                device=self.device
            )

    def stream_video_tracking(
        self,
        video_path: str,
        vid_stride: int = 1,
        include_annotated_frame: bool = False,
        show_labels: bool = True,
        line_width: int = 2,
        fallback_frame_rate: int = 30
    ) -> Iterator[FrameTrackingResult]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_width <= 0 or frame_height <= 0:
                raise ValueError("Could not read valid frame dimensions from video.")

            effective_fps = int(round(native_fps)) if native_fps and native_fps > 0 else fallback_frame_rate
            effective_fps = max(1, effective_fps)

            self._initialize_tracker(effective_fps)

            frame_num = 0
            processed_count = 0
            start_time = time.time()

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_num += 1

                if vid_stride > 1 and (frame_num - 1) % vid_stride != 0:
                    continue

                processed_count += 1
                frame_start_time = time.time()
                
                tracked_objects = self.tracker_wrapper.process_frame(frame)
                
                frame_processing_time = time.time() - frame_start_time
                
                timestamp = frame_num / effective_fps

                annotated_frame = None
                if include_annotated_frame:
                    annotated_frame = self.tracker_wrapper.annotate_frame(
                        frame, tracked_objects, show_labels, line_width
                    )

                result = FrameTrackingResult(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    tracked_objects=tracked_objects,
                    frame_shape=(frame_height, frame_width),
                    processing_time=frame_processing_time,
                    annotated_frame=annotated_frame
                )

                yield result

        finally:
            cap.release()

    def stream_camera_tracking(
        self,
        camera_index: int = 0,
        include_annotated_frame: bool = False,
        show_labels: bool = True,
        line_width: int = 2,
        frame_rate: int = 30
    ) -> Iterator[FrameTrackingResult]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Could not open camera with index: {camera_index}")

        try:
            cap.set(cv2.CAP_PROP_FPS, frame_rate)
            
            self._initialize_tracker(frame_rate)

            frame_num = 0
            start_time = time.time()

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_num += 1
                frame_start_time = time.time()
                
                tracked_objects = self.tracker_wrapper.process_frame(frame)
                
                frame_processing_time = time.time() - frame_start_time
                
                timestamp = time.time() - start_time

                frame_height, frame_width = frame.shape[:2]

                annotated_frame = None
                if include_annotated_frame:
                    annotated_frame = self.tracker_wrapper.annotate_frame(
                        frame, tracked_objects, show_labels, line_width
                    )

                result = FrameTrackingResult(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    tracked_objects=tracked_objects,
                    frame_shape=(frame_height, frame_width),
                    processing_time=frame_processing_time,
                    annotated_frame=annotated_frame
                )

                yield result

        finally:
            cap.release()

    def reset_tracker(self):
        if self.tracker_wrapper:
            self.tracker_wrapper.reset_tracker()


def track_video_sahi(
    video_path: str,
    output_path: str,
    model_path: str,
    tracker_type: str = 'botsort',
    tracker_config_path: Optional[str] = None,
    fallback_frame_rate: int = 30,
    window_size_ratio: Tuple[float, float] = (0.7, 0.7),
    overlap_ratio: Tuple[float, float] = (0.2, 0.2),
    img_size: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    nms_global: float = 0.5,
    show_labels: bool = True,
    line_width: int = 2,
    show_preview: bool = False,
    vid_stride: int = 1,
    classes: Optional[List[int]] = None,
    output_fps_override: Optional[int] = None
) -> Optional[str]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stream_tracker = VideoStreamTracker(
        model_path=model_path,
        tracker_type=tracker_type,
        tracker_config_path=tracker_config_path,
        window_size_ratio=window_size_ratio,
        overlap_ratio=overlap_ratio,
        img_size=img_size,
        conf=conf,
        iou=iou,
        nms_global=nms_global,
        classes=classes
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if output_fps_override and output_fps_override > 0:
        effective_fps = output_fps_override
    elif native_fps and native_fps > 0:
        effective_fps = int(round(native_fps))
    else:
        effective_fps = fallback_frame_rate

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError(f"Could not create video writer at {output_path}")

    try:
        processed_count = 0
        start_time = time.time()
        preview_window_name = "SAHI Tracking Preview"

        for result in stream_tracker.stream_video_tracking(
            video_path=video_path,
            vid_stride=vid_stride,
            include_annotated_frame=True,
            show_labels=show_labels,
            line_width=line_width
        ):
            processed_count += 1
            
            out.write(result.annotated_frame)
            
            if show_preview:
                cv2.imshow(preview_window_name, result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if processed_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed if elapsed > 0 else 0
                eta = ((total_frames - result.frame_number) / vid_stride / fps_proc) if fps_proc > 0 else 0
                eta_str = f"{eta:.1f}s" if eta > 0 else "N/A"
                print(f"Frame {result.frame_number}/{total_frames} | Processed {processed_count} | "
                      f"FPS: {fps_proc:.2f} | ETA: {eta_str}   ", end='\r')

    except Exception as e:
        log.error(f"\n--- An error occurred during processing ---")
        log.error(f"Error type: {type(e).__name__}")
        log.error(f"Error details: {e}")
        import traceback
        traceback.print_exc()

    finally:
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0

        log.info("\n" + "-" * 40)
        log.info("--- Video Processing Finished ---")
        log.info(f"Total frames processed: {processed_count}")
        log.info(f"Total processing time: {total_time:.2f} seconds")
        log.info(f"Average processing FPS: {avg_fps:.2f}")
        log.info(f"Output video saved to: {output_path}")
        log.info("-" * 40)

    return output_path


if __name__ == "__main__":
    VIDEO_INPUT_PATH = "/home/student/projects/RusTitW/data/detection/video_check/check.mp4"
    MODEL_WEIGHTS_PATH = "/home/student/projects/RusTitW/detection_models/yolo12m_v3/weights/last.engine"
    OUTPUT_DIR = "/home/student/projects/RusTitW/data/detection/video_check/"
    OUTPUT_FILENAME = "sahi_streaming_output.mp4"

    stream_tracker = VideoStreamTracker(
        model_path=MODEL_WEIGHTS_PATH,
        tracker_type='botsort',
        window_size_ratio=(0.7, 0.7),
        overlap_ratio=(0.1, 0.1),
        img_size=640,
        conf=0.10,
        iou=0.10,
        nms_global=0.10,
        classes=[0]
    )

    log.info("Starting streaming video processing...")
    
    for result in stream_tracker.stream_video_tracking(
        video_path=VIDEO_INPUT_PATH,
        vid_stride=1,
        include_annotated_frame=False
    ):
        log.info(f"Frame {result.frame_number}: {len(result.tracked_objects)} objects tracked "
              f"(processing time: {result.processing_time:.3f}s)")
        
        for obj in result.tracked_objects:
            log.info(f"  - ID:{obj['track_id']} {obj['class_name']} conf:{obj['confidence']:.2f}")
        
        if result.frame_number > 100:
            break

    log.info("Streaming processing completed.")
