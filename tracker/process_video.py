import os
import cv2
import numpy as np
import torch
import torchvision
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from ultralytics import YOLO # RTDETR might need different handling if used
from ultralytics.trackers import BYTETracker, BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
# Removed: from ultralytics.engine.results import Boxes - Will create a mock structure instead
from pathlib import Path
import types # To create simple namespaces easily

# Define colors for drawing
_colors_list = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (192, 192, 192), (128, 128, 128), (255, 165, 0), (255, 192, 203)
]

def get_color(index: int) -> Tuple[int, int, int]:
    """Gets a consistent color for a given index."""
    return _colors_list[index % len(_colors_list)]

# --- Minimal Mock Structures for Tracker Input ---
# The tracker's update(results, img) expects results.boxes.data
# where .data is a tensor like [x1, y1, x2, y2, conf, cls]

class MockBoxesData:
    """Holds the raw detection tensor and original shape."""
    def __init__(self, boxes_data_np: np.ndarray, orig_shape: Tuple[int, int]):
        if boxes_data_np.ndim == 1 and boxes_data_np.shape[0] == 6:
            boxes_data_np = boxes_data_np[np.newaxis, :]
        elif boxes_data_np.shape[0] == 0:
            boxes_data_np = np.empty((0, 6), dtype=np.float32)

        self.data = torch.from_numpy(boxes_data_np).float().cpu()
        self.orig_shape = orig_shape

        # --- Derive properties from .data ---
        if self.data.numel() > 0:
            self.xyxy = self.data[:, :4]
            self.conf = self.data[:, 4]
            self.cls = self.data[:, 5].int()
            # --- ADD XYWH CALCULATION ---
            x1, y1, x2, y2 = self.xyxy.unbind(1)
            self.xywh = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)
            # --- END ADDITION ---
        else: # Handle empty case
            self.xyxy = torch.empty((0, 4), device=self.data.device)
            self.conf = torch.empty(0, device=self.data.device)
            self.cls = torch.empty(0, dtype=torch.int, device=self.data.device)
            # --- ADD EMPTY XYWH ---
            self.xywh = torch.empty((0, 4), device=self.data.device)
            # --- END ADDITION ---

    def cpu(self):
        # Already on CPU, but method might be called
        return self

    def __len__(self):
        return self.data.shape[0]


class MockResults:
    """Holds a MockBoxesData instance under the .boxes attribute."""
    def __init__(self, boxes_data_np: np.ndarray, orig_shape: Tuple[int, int]):
        self.boxes = MockBoxesData(boxes_data_np, orig_shape)
        # --- Expose required attributes directly ---
        self.conf = self.boxes.conf
        self.xywh = self.boxes.xywh # <-- ADD THIS LINE
        self.cls = self.boxes.cls
        # --- End Exposure ---

        # Add other attributes if the tracker expects them (e.g., names, masks)
        self.names = {} # Add dummy names if needed, tracker might not use it
        self.masks = None # Add dummy masks if needed
        self.probs = None # Add dummy probs if needed
        self.keypoints = None # Add dummy keypoints if needed
        self.orig_shape = orig_shape
        self.orig_img = None # Can be set if tracker uses it

    def cpu(self):
        # Delegate to boxes if needed, but usually not called on Results directly
        self.boxes.cpu()
        return self

    def __len__(self):
        # Usually length of results means number of boxes
        return len(self.boxes)


# --- SAHI-like Tracking Wrapper ---
class SAHITrackingWrapper:
    def __init__(
        self,
        model_path: str,
        tracker_type: str = 'bytetrack',
        tracker_config_path: Optional[str] = None,
        frame_rate: int = 30,
        window_size_ratio: Tuple[float, float] = (0.5, 0.5),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45, # NMS inside model.predict
        nms_threshold_global: float = 0.5, # NMS for merging windows
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading detection model from {model_path}...")
        # Initialize model - device specified during predict, not here for exported models
        self.model = YOLO(model_path, task='detect')
        self.model_names = self.model.names # Get class names
        print("Detection model loaded.")

        # --- Tracker Config Loading ---
        script_dir = Path(__file__).parent.resolve()
        default_cfg_path = script_dir / 'cfg' / 'trackers' / f'{tracker_type}.yaml'

        if tracker_config_path and os.path.exists(tracker_config_path):
            tracker_config_file = tracker_config_path
            print(f"Using custom tracker config: {tracker_config_file}")
        elif default_cfg_path.exists():
            tracker_config_file = str(default_cfg_path)
            print(f"Using default tracker config relative to script: {tracker_config_file}")
        else:
            try:
                # Fallback: look relative to CWD or where ultralytics installs defaults
                tracker_config_file = check_yaml(f'{tracker_type}.yaml')
                print(f"Using ultralytics default tracker config found via check_yaml: {tracker_config_file}")
            except FileNotFoundError as e:
                 raise FileNotFoundError(f"Could not find tracker config: '{tracker_type}.yaml'. "
                                         f"Checked relative path, CWD, and ultralytics defaults. "
                                         f"Provide a valid 'tracker_config_path'.") from e

        tracker_cfg_dict = YAML.load(str(tracker_config_file))
        if tracker_cfg_dict.get('tracker_type') != tracker_type:
            print(f"Warning: Tracker config file specifies type '{tracker_cfg_dict.get('tracker_type')}', "
                  f"but '{tracker_type}' was requested. Overriding config type to '{tracker_type}'.")
            tracker_cfg_dict['tracker_type'] = tracker_type
        tracker_cfg = IterableSimpleNamespace(**tracker_cfg_dict)
        # --- End Tracker Config Loading ---


        if tracker_cfg.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError(f"Unsupported tracker type: {tracker_cfg.tracker_type}")

        if tracker_type == 'botsort' and getattr(tracker_cfg, 'with_reid', False):
            print("Warning: BoT-SORT with ReID features are not explicitly handled by this SAHI wrapper.")

        print(f"Initializing {tracker_cfg.tracker_type} tracker (frame_rate={frame_rate})...")
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](args=tracker_cfg, frame_rate=frame_rate)
        print("Tracker initialized.")

        self.window_size_ratio = window_size_ratio
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold_global = nms_threshold_global
        self.classes = classes


    def _get_windows(self, frame_width: int, frame_height: int) -> List[Tuple[int, int, int, int]]:
        # Same as previous version, seems robust
        win_w = max(1, int(frame_width * self.window_size_ratio[0]))
        win_h = max(1, int(frame_height * self.window_size_ratio[1]))
        overlap_w = int(win_w * self.overlap_ratio[0])
        overlap_h = int(win_h * self.overlap_ratio[1])

        step_w = max(1, win_w - overlap_w)
        step_h = max(1, win_h - overlap_h)

        windows = []
        for y1 in range(0, frame_height, step_h):
            y2 = min(y1 + win_h, frame_height)
            current_y1 = y1 if (y1 + win_h <= frame_height) else max(0, frame_height - win_h) # Adjust last row start
            current_y2 = current_y1 + win_h

            for x1 in range(0, frame_width, step_w):
                x2 = min(x1 + win_w, frame_width)
                current_x1 = x1 if (x1 + win_w <= frame_width) else max(0, frame_width - win_w) # Adjust last col start
                current_x2 = current_x1 + win_w

                window_coords = (current_x1, current_y1, current_x2, current_y2)
                if window_coords not in windows and current_x2 > current_x1 and current_y2 > current_y1:
                    windows.append(window_coords)

                if x2 == frame_width: break # Reached right edge
            if y2 == frame_height: break # Reached bottom edge

        return windows

    def _nms_global(self, detections: List[Tuple[float, float, float, float, float, int]]) -> np.ndarray:
        # Same as previous version, uses torchvision NMS
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        # Use torch for NMS. Place temporary tensors on the primary device for speed.
        device_for_nms = self.device if torch.cuda.is_available() else torch.device('cpu')
        detections_tensor = torch.tensor(detections, dtype=torch.float32, device=device_for_nms)

        if detections_tensor.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = detections_tensor[:, :4] # xyxy
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
        # Return as CPU NumPy array, expected format [x1, y1, x2, y2, conf, cls]
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
        
        # Подготовка всех окон для одновременной обработки
        for x1, y1, x2, y2 in windows:
            window_img = frame[y1:y2, x1:x2]
            if window_img.size < 4:
                continue
                
            window_coords.append((x1, y1, x2, y2))
            
            # Изменение размера если нужно
            if self.img_size != (window_img.shape[1], window_img.shape[0]):
                window_img = cv2.resize(window_img, self.img_size)
                
            batch_images.append(window_img)
        
        # Проверка на количество окон и дополнение до batch_size при необходимости
        num_real_windows = len(batch_images)
        
        if num_real_windows == 0:
            return []
        
        # Дополнение батча до фиксированного размера, если окон меньше batch_size
        if num_real_windows < batch_size:
            zero_shape = batch_images[0].shape
            zero_img = np.zeros(zero_shape, dtype=np.uint8)
            
            # Дополняем батч пустыми изображениями до batch_size
            batch_images.extend([zero_img.copy() for _ in range(batch_size - num_real_windows)])
        
        try:
            # Преобразование списка изображений в numpy-массив для батча
            batch_input = np.array(batch_images)
            
            # Единственный проход через модель с полным батчем
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
            
            # Обработка результатов только для реальных окон (игнорирование паддинга)
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
            print(f"Warning: Error processing batch: {type(e).__name__}: {str(e)}")

        # 2. Merge detections using Global NMS
        merged_detections_np = self._nms_global(all_detections_in_frame_coords)
        # merged_detections_np has shape (N, 6) with [x1, y1, x2, y2, conf, cls]

        # 3. Prepare input for the tracker using MockResults
        mock_results_for_tracker = MockResults(merged_detections_np, (frame_height, frame_width))
        # Set orig_img on the mock results if the tracker might need it (some advanced features might)
        # mock_results_for_tracker.orig_img = frame # Uncomment if needed

        # 4. Update the tracker
        try:
            # Pass the mock object that adheres to the expected structure
            # Tracker expects results.boxes.data for BYTETracker/BoTSORT
            tracked_output_np = self.tracker.update(mock_results_for_tracker, frame)
        except Exception as e:
             # print(f"Error during tracker update: {type(e).__name__}: {e}")
             import traceback
             # traceback.print_exc()
             return [] # Return empty list if tracker update fails

        # 5. Format tracker output (modify this section)
        tracked_objects_list = []
        if isinstance(tracked_output_np, np.ndarray) and tracked_output_np.size > 0:
            # Трекер (BOT-SORT) обычно возвращает [x1, y1, x2, y2, track_id, conf, cls, idx] (8 столбцов)
            # В некоторых случаях может быть 7 (без idx) или 9 (если с углом)
            output_cols = tracked_output_np.shape[1]

            # Обработка 8 столбцов: [x1, y1, x2, y2, track_id, conf, cls, idx]
            if output_cols == 8:
                for row in tracked_output_np:
                    x1, y1, x2, y2, track_id, conf, cls_id, idx = row # Парсим все 8 значений
                    # Преобразование типов
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id, cls_id = int(track_id), int(cls_id) # idx можно оставить float или int(idx) если нужно

                    # Проверка корректности координат
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)

                    if x2 > x1 and y2 > y1:
                        class_name = self.model_names.get(cls_id, f"Cls_{cls_id}")
                        tracked_objects_list.append({
                            'box': [x1, y1, x2, y2], # Bbox в формате [x1, y1, x2, y2]
                            'track_id': track_id,
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'original_det_idx': int(idx) # Можно добавить оригинальный индекс детекции
                        })

            # Обработка 7 столбцов (если вдруг трекер вернет старый формат)
            elif output_cols == 7:
                 print("Warning: Tracker output shape is (N, 7). Processing as [x1, y1, x2, y2, track_id, conf, cls].")
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

            # Опционально: Обработка 9 столбцов, если включено отслеживание угла (xywha + ...)
            elif output_cols == 9:
                LOGGER.warning("Tracker output shape is (N, 9) - likely xywha + tracking info. Processing as xyxy for drawing.")
                for row in tracked_output_np:
                    # Предполагаем формат: [cx, cy, w, h, angle, track_id, conf, cls, idx]
                    cx, cy, w, h, angle, track_id, conf, cls_id, idx = row
                    # Конвертируем xywh в xyxy для отрисовки
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
                            'box': [x1, y1, x2, y2], # Конвертируем в xyxy для отрисовки
                            'track_id': track_id,
                            'class_id': cls_id,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'original_det_idx': int(idx),
                            'angle': float(angle) # Сохраняем угол, если нужен
                        })


            else:
                # Оригинальное предупреждение для других неожиданных форм
                print(f"Warning: Tracker output NumPy array has unexpected shape {tracked_output_np.shape}. Expected 7, 8, or 9 columns.")

        elif isinstance(tracked_output_np, (np.ndarray, list)) and len(tracked_output_np) == 0:
            pass # No tracks returned
        else:
            print(f"Warning: Tracker returned unexpected output type: {type(tracked_output_np)}")

        return tracked_objects_list


    def reset_tracker(self):
        """Resets the internal state of the tracker."""
        print("Resetting tracker state...")
        if hasattr(self.tracker, 'reset'):
             self.tracker.reset()
        else:
             print("Warning: Tracker object does not have a .reset() method.")


# --- Video Processing Function ---
# (track_video_sahi function remains largely the same as the previous version,
# only the call to the wrapper's method name changes if you renamed it,
# and the initialization error handling might be slightly different)

def track_video_sahi(
    video_path: str,
    output_path: str,
    model_path: str,
    tracker_type: str = 'bytetrack',
    tracker_config_path: Optional[str] = None,
    fallback_frame_rate: int = 30,
    window_size_ratio: Tuple[float, float] = (0.5, 0.5),
    overlap_ratio: Tuple[float, float] = (0.2, 0.2),
    img_size: int = 640,
    conf: float = 0.25,
    iou: float = 0.45, # NMS inside model.predict
    nms_global: float = 0.5, # NMS for merging windows
    show_labels: bool = True,
    line_width: int = 2,
    show_preview: bool = False,
    vid_stride: int = 1,
    classes: Optional[List[int]] = None,
    output_fps_override: Optional[int] = None
) -> Optional[str]:

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise ValueError("Could not read valid frame dimensions from video.")

    # Determine FPS
    if output_fps_override and output_fps_override > 0:
         effective_fps = output_fps_override
         print(f"Using overridden output FPS: {effective_fps}")
    elif native_fps and native_fps > 0:
        effective_fps = int(round(native_fps))
        print(f"Using native video FPS: {native_fps:.2f} (rounded to {effective_fps})")
    else:
        effective_fps = fallback_frame_rate
        print(f"Warning: Could not read native FPS or override not set. Using fallback: {effective_fps}")
    effective_fps = max(1, effective_fps) # Ensure FPS is at least 1


    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Could not create video writer at {output_path}")

    # Initialize SAHI Tracking Wrapper
    tracker_wrapper = None # Define before try block
    try:
        tracker_wrapper = SAHITrackingWrapper(
            model_path=model_path,
            tracker_type=tracker_type,
            tracker_config_path=tracker_config_path,
            frame_rate=effective_fps,
            window_size_ratio=window_size_ratio,
            overlap_ratio=overlap_ratio,
            img_size=img_size,
            conf_threshold=conf,
            iou_threshold=iou,
            nms_threshold_global=nms_global,
            classes=classes,
            device='cuda:0'
        )
    except Exception as e:
        print(f"--- Failed to initialize SAHI Tracking Wrapper ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        print("-" * 40)
        cap.release()
        out.release()
        return None

    # --- Main Processing Loop ---
    frame_num = 0
    processed_count = 0
    start_time = time.time()
    preview_window_name = "SAHI Tracking Preview"

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_num += 1

            if vid_stride > 1 and (frame_num - 1) % vid_stride != 0:
                if show_preview:
                     preview_frame = frame.copy()
                     cv2.putText(preview_frame, f"Skipped Frame {frame_num}", (20, 40),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                     cv2.imshow(preview_window_name, preview_frame)
                     if cv2.waitKey(1) & 0xFF == ord('q'): break
                out.write(frame)
                continue

            # Process the frame
            processed_count += 1
            tracked_objects = tracker_wrapper.process_frame(frame)
            annotated_frame = frame.copy()

            # Draw results (same drawing logic as before)
            for obj in tracked_objects:
                box = obj['box']; track_id = obj['track_id']; class_name = obj['class_name']
                confidence = obj['confidence']; x1, y1, x2, y2 = map(int, box)
                color = get_color(track_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_width)
                if show_labels:
                    label_text = f"ID:{track_id} {class_name} ({confidence:.2f})"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    label_y = max(y1 - 10, text_h + baseline + 3) # Adjust baseline calc
                    label_x = max(x1, 0)
                    cv2.rectangle(annotated_frame, (label_x, label_y - text_h - baseline),
                                  (label_x + text_w, label_y), color, -1)
                    cv2.putText(annotated_frame, label_text, (label_x, label_y - (baseline//2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(annotated_frame)
            if show_preview:
                cv2.imshow(preview_window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Print progress
            if processed_count % 50 == 0: # Print more often
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed if elapsed > 0 else 0
                eta = ((total_frames - frame_num) / (vid_stride if vid_stride > 0 else 1) / fps_proc) if fps_proc > 0 else 0
                eta_str = f"{eta:.1f}s" if eta > 0 else "N/A"
                print(f"Frame {frame_num}/{total_frames} | Processed {processed_count} | "
                      f"FPS: {fps_proc:.2f} | ETA: {eta_str}   ", end='\r')


    except Exception as e:
        print(f"\n--- An error occurred during processing ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        print("--- Attempting to finalize video ---")

    finally:
        # Release resources
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0

        print("\n" + "-" * 40) # Newline before final stats
        print("--- Video Processing Finished ---")
        print(f"Total frames in video: {total_frames}")
        print(f"Frames processed (stride={vid_stride}): {processed_count}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average processing FPS: {avg_fps:.2f}")
        print(f"Output video saved to: {output_path}")
        print("-" * 40)

    return output_path


# --- Example Usage ---
if __name__ == "__main__":

    # --- Configuration ---
    VIDEO_INPUT_PATH = "/home/student/projects/RusTitW/data/detection/video_check/check.mp4"
    MODEL_WEIGHTS_PATH = "/home/student/projects/RusTitW/detection_models/yolo12m_v3/weights/last.engine"
    OUTPUT_DIR = "/home/student/projects/RusTitW/data/detection/video_check/"
    OUTPUT_FILENAME = "sahi_refactored_output_v3.mp4"
    TRACKER_CONFIG = "/home/student/projects/RusTitW/notebooks/botsort.yaml" # Or path to bytetrack.yaml, or None
    TRACKER_TYPE_TO_USE = 'botsort'   # Ensure this matches the config or desired tracker

    SAHI_WINDOW_SIZE = (0.7, 0.7)
    SAHI_OVERLAP = (0.1, 0.1)
    GLOBAL_NMS_IOU = 0.10
    MODEL_INPUT_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.10
    IOU_THRESHOLD = 0.10 # NMS inside model.predict (less critical than global)
    PROCESS_EVERY_N_FRAMES = 1
    SHOW_PREVIEW_WINDOW = False
    DRAW_LABELS = True
    BOX_LINE_WIDTH = 2
    CLASSES_TO_TRACK = [0] # None to track all

    # --- End Configuration ---

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    if not os.path.exists(VIDEO_INPUT_PATH):
         print(f"Error: Input video not found at '{VIDEO_INPUT_PATH}'")
    elif not os.path.exists(MODEL_WEIGHTS_PATH):
         print(f"Error: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
    elif TRACKER_CONFIG and not os.path.exists(TRACKER_CONFIG):
         print(f"Error: Custom tracker config specified but not found at '{TRACKER_CONFIG}'")
    else:
        print("\n--- Starting SAHI-like Video Tracking ---")
        print(f"Input Video: {VIDEO_INPUT_PATH}")
        print(f"Output Video: {video_output_path}")
        print(f"Model: {MODEL_WEIGHTS_PATH}")
        print(f"Tracker: {TRACKER_TYPE_TO_USE}")
        print(f"Tracker Config: {'Default/Auto-Detect' if TRACKER_CONFIG is None else TRACKER_CONFIG}")
        print(f"SAHI Window Ratio: {SAHI_WINDOW_SIZE}, Overlap: {SAHI_OVERLAP}")
        print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Global NMS Threshold: {GLOBAL_NMS_IOU}")
        print(f"Classes Filter: {'All' if CLASSES_TO_TRACK is None else CLASSES_TO_TRACK}")
        print("-" * 40)

        try:
            final_output_file = track_video_sahi(
                video_path=VIDEO_INPUT_PATH,
                output_path=video_output_path,
                model_path=MODEL_WEIGHTS_PATH,
                tracker_type=TRACKER_TYPE_TO_USE,
                tracker_config_path=None,
                window_size_ratio=SAHI_WINDOW_SIZE,
                overlap_ratio=SAHI_OVERLAP,
                img_size=MODEL_INPUT_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                nms_global=GLOBAL_NMS_IOU,
                show_labels=DRAW_LABELS,
                line_width=BOX_LINE_WIDTH,
                show_preview=SHOW_PREVIEW_WINDOW,
                vid_stride=PROCESS_EVERY_N_FRAMES,
                classes=CLASSES_TO_TRACK
            )

            if final_output_file:
                print(f"Script finished successfully.") # Success message at the very end
            else:
                print("\nTracking process failed or was interrupted before completion.")

        except FileNotFoundError as e:
             print(f"\nError: A required file was not found.")
             print(e)
        except Exception as e:
             print(f"\nAn unexpected error occurred:")
             import traceback
             traceback.print_exc()