import os
import asyncio
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import numpy as np
from src.triton_api.main_endpoint import MainEndpointClient
from src.triton_api.stream_endpoint import StreamEndpointClient
from src.triton_api.websocket_endpoint import WebSocketEndpointClient
from src.scripts.tracker import VideoStreamTracker, FrameTrackingResult

from src.utils.custom_logging import setup_logging

log = setup_logging()


class DonutInferenceClient:
    def __init__(
        self,
        detection_source: str,
        model_name: str,
        triton_main_url: Optional[str] = None,
        triton_stream_url: Optional[str] = None,
        triton_ws_url: Optional[str] = None
    ) -> None:
        self.detection_source = detection_source
        self.model_name = model_name
        self._main_url = triton_main_url
        self._stream_url = triton_stream_url
        self._ws_url = triton_ws_url

    async def infer(self, image: np.ndarray) -> str:
        try:
            log.info(f"Detection source: {self.detection_source}, URLs: main={self._main_url}, stream={self._stream_url}, ws={self._ws_url}")
            if self.detection_source == "main" and self._main_url:
                async with MainEndpointClient(base_url=self._main_url) as client:
                    resp = await client.donut_inference_from_arrays(
                        image_arrays=[image], filenames=["donut.jpg"]
                    )
            elif self.detection_source == "stream" and self._stream_url:
                async with StreamEndpointClient(base_url=self._stream_url, chunk_size=1) as client:
                    resp = await client.stream_collect_from_arrays(
                        image_arrays=[image], model_name=self.model_name
                    )
            elif self.detection_source == "ws" and self._ws_url:
                client = WebSocketEndpointClient(base_url=self._ws_url)
                resp = await client.run_inference_session(
                    model_name=self.model_name, images=[image]
                )
            else:
                log.error(f"Unsupported detection source: {self.detection_source}")
                raise ValueError(f"Unsupported detection source: {self.detection_source}")
            log.info(f"Received response: {resp}")
            if isinstance(resp, list):
                first_chunk = resp[0]
                results = first_chunk.get("results", [])
            else:
                results = resp.get("results", [])
            if not results:
                return ""
            first = results[0]
            return first.get("text") or first.get("text_sequence", "") or ""
        except Exception as e:
            log.error(f"Donut inference error: {e}\n{traceback.format_exc()}")
            return ""


class DonutTextRecognizer:
    def __init__(
        self,
        video_tracker: VideoStreamTracker,
        inference_client: DonutInferenceClient,
        history_length: int = 8,
        min_crop_size: int = 16
    ) -> None:
        self._tracker = video_tracker
        self._client = inference_client
        self._history_length = max(1, history_length)
        self._min_crop_size = min_crop_size
        self._buffers: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._tasks: Dict[int, asyncio.Task] = {}
        self._results: Dict[int, str] = {}

    async def recognize_text_from_tracking_stream(
        self,
        tracking_stream: AsyncIterator[FrameTrackingResult]
    ) -> AsyncIterator[FrameTrackingResult]:
        async for frame in tracking_stream:
            img = frame.annotated_frame
            for obj in frame.tracked_objects:
                tid = obj["track_id"]
                obj["recognized_text"] = None
                if tid in self._results:
                    obj["recognized_text"] = self._results[tid]
                    continue
                if img is None:
                    continue
                x1, y1, x2, y2 = obj["box"]
                w, h = x2 - x1, y2 - y1
                if w < self._min_crop_size or h < self._min_crop_size:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    log.warning(f"Empty crop for track {tid}: box={x1, y1, x2, y2}")
                    continue
                self._buffers[tid].append(crop)
                log.debug(f"Appended crop for tid={tid}, buffer now {len(self._buffers[tid])}")
                if len(self._buffers[tid]) >= self._history_length:
                    mid = self._history_length // 2
                    mid_frame = self._buffers[tid][mid]
                    log.debug(
                        f"Scheduling Donut inference for track {tid}, buffer size={len(self._buffers[tid])}, mid_frame shape={mid_frame.shape}"
                    )
                    text = await self._client.infer(mid_frame)
                    self._results[tid] = text
                    obj["recognized_text"] = text
            yield frame


async def recognize_text_from_video(
    video_path: str,
    model_path: Optional[str],
    tracker_type: str,
    tracker_config_path: Optional[str],
    window_size_ratio: Tuple[float, float],
    overlap_ratio: Tuple[float, float],
    img_size: Union[int, Tuple[int, int]],
    conf: float,
    iou: float,
    nms_global: float,
    classes: Optional[List[int]],
    device: Optional[Union[str, Any]],
    tracker_detection_source: str,
    triton_stream_url: Optional[str] = None,
    triton_ws_url: Optional[str] = None,
    triton_batch_url: Optional[str] = None,
    triton_model_name: str = "yolo",
    triton_chunk_size: int = 1,
    donut_detection_source: str = "main",
    donut_triton_main_url: Optional[str] = None,
    donut_triton_stream_url: Optional[str] = None,
    donut_triton_ws_url: Optional[str] = None,
    donut_model_name: str = "donut",
    history_length: int = 8
) -> AsyncIterator[FrameTrackingResult]:

    tracker = VideoStreamTracker(
        model_path=model_path,
        tracker_type=tracker_type,
        tracker_config_path=tracker_config_path,
        window_size_ratio=window_size_ratio,
        overlap_ratio=overlap_ratio,
        img_size=img_size,
        conf=conf,
        iou=iou,
        nms_global=nms_global,
        classes=classes,
        device=device,
        detection_source=tracker_detection_source,
        triton_stream_url=triton_stream_url,
        triton_ws_url=triton_ws_url,
        triton_batch_url=triton_batch_url,
        triton_model_name=triton_model_name,
        triton_chunk_size=triton_chunk_size
    )
    tracking_stream = tracker.stream_video_tracking(
        video_path=video_path,
        include_annotated_frame=True,
        show_labels=True
    )
    client = DonutInferenceClient(
        detection_source=donut_detection_source,
        model_name=donut_model_name,
        triton_main_url=donut_triton_main_url,
        triton_stream_url=donut_triton_stream_url,
        triton_ws_url=donut_triton_ws_url
    )
    recognizer = DonutTextRecognizer(tracker, client, history_length)
    async for enriched in recognizer.recognize_text_from_tracking_stream(tracking_stream):
        yield enriched


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).parent.parent.resolve()
    docs = project_root.parent / "docs"
    video_file = docs / "check.mp4"
    if not video_file.exists():
        logger.error(f"{video_file} not found")
        return

    triton_http = os.getenv("TRITON_API_URL", "http://localhost:8000")
    triton_ws = os.getenv("TRITON_WS_URL", "ws://localhost:8000")
    scenarios = [
        ("donut_main", dict(donut_detection_source="main", donut_triton_main_url=triton_http)),
        # ("donut_stream", dict(donut_detection_source="stream", donut_triton_stream_url=triton_http)),
        # ("donut_ws", dict(donut_detection_source="ws", donut_triton_ws_url=triton_ws)),
    ]

    for name, ds in scenarios:
        logger.info(f"--- {name} ---")
        async for frame in recognize_text_from_video(
            video_path=str(video_file),
            model_path=None, # str(docs / "last.engine"),
            tracker_type="botsort",
            tracker_config_path=None,
            window_size_ratio=(0.7, 0.7),
            overlap_ratio=(0.1, 0.1),
            img_size=640,
            conf=0.1,
            iou=0.1,
            nms_global=0.1,
            classes=[0],
            device=None,
            tracker_detection_source="triton_batch",
            triton_batch_url=triton_http,
            **ds,
            history_length=24
        ):
            parts = []
            for obj in frame.tracked_objects:
                t = obj.get("recognized_text")
                parts.append(f"ID={obj['track_id']} text={t!r}")
            log.info(f"Frame {frame.frame_number}: " + "; ".join(parts))


if __name__ == "__main__":
    asyncio.run(main())