import subprocess
import re
import os
import tempfile
import asyncio
from pathlib import Path
from typing import Tuple, Optional
import json
import signal

from src.utils.custom_logging import get_logger
from src.scripts.cancel_handler import CancellationHandler

log = get_logger(__name__)


class VideoDownloader:
    def __init__(self, 
                 min_duration: int = 1, 
                 max_duration: int = 10,
                 quality: str = 'best',
                 frame_dimensions: Tuple[int, int] = (64, 64)
                 ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.quality = quality
        self.frame_dimensions = frame_dimensions

    def download_video_to_temp(self, url: str) -> Optional[str]:
        cancellation = CancellationHandler()

        try:
            cancellation.check_cancellation()

            if not self._is_video_valid(url):
                log.warning(f"Video {url} does not meet the criteria. Skipping download.")
                return None
            
            cancellation.check_cancellation()

            temp_dir = tempfile.mkdtemp(prefix="fr_video_")
            output_pattern = "%(id)s.%(ext)s"

            command = [
                'yt-dlp',
                '--format', self.quality,
                '--output', str(Path(temp_dir) / output_pattern),
                '--no-warnings',
                url
            ]
            
            try:
                log.info(f"Attempting to download video from {url}")
                
                result = subprocess.run(
                    command, 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    timeout=120
                )
                
                downloaded_files = list(Path(temp_dir).glob("*"))
                if not downloaded_files:
                    log.error(f"No files found after download in {temp_dir}")
                    self._cleanup_temp_dir(temp_dir)
                    return None

                video_file = downloaded_files[0]
                if video_file.stat().st_size == 0:
                    log.error(f"Downloaded video file is empty: {video_file}")
                    self._cleanup_temp_dir(temp_dir)
                    return None

                log.info(f"Successfully downloaded video to {video_file}")
                return str(video_file)
                
            except subprocess.TimeoutExpired:
                log.error(f"Download timeout for {url}")
                self._cleanup_temp_dir(temp_dir)
                return None
            except subprocess.CalledProcessError as e:
                if e.returncode == -signal.SIGINT:
                    log.info("Download cancelled by user signal")
                    raise asyncio.CancelledError("Download cancelled")
                log.error(f"yt-dlp failed for {url}. Error: {e.stderr}")
                self._cleanup_temp_dir(temp_dir)
                return None
                
        except asyncio.CancelledError:
            log.info("Video download cancelled")
            raise
        except Exception as e:
            log.error(f"Unexpected error downloading {url}: {e}")
            return None

    def _is_video_valid(self, url: str) -> bool:
        cancellation = CancellationHandler()
        
        try:
            cancellation.check_cancellation()
            
            duration_result = subprocess.run(
                ['yt-dlp', '--get-duration', '--no-warnings', url],
                capture_output=True, 
                text=True, 
                timeout=30,
                check=True
            )
            
            duration_str = duration_result.stdout.strip()
            log.debug(f"Duration for {url}: {duration_str}")

            if not self._is_valid_duration_format(duration_str):
                log.warning(f"Invalid duration format: {duration_str}")
                return False

            duration_minutes = self._convert_duration_to_minutes(duration_str)
            if not (self.min_duration <= duration_minutes <= self.max_duration):
                log.warning(f"Duration {duration_minutes:.2f}min not in range [{self.min_duration}, {self.max_duration}]")
                return False

            cancellation.check_cancellation()

            info_result = subprocess.run(
                ['yt-dlp', '--dump-json', '--no-warnings', url],
                capture_output=True, 
                text=True, 
                timeout=30,
                check=True
            )
            
            video_info = json.loads(info_result.stdout)
            return self._check_video_resolution(video_info, url)

        except subprocess.TimeoutExpired:
            log.error(f"Validation timeout for {url}")
            return False
        except subprocess.CalledProcessError as e:
            if e.returncode == -signal.SIGINT:
                raise asyncio.CancelledError("Validation cancelled")
            log.error(f"yt-dlp validation failed for {url}: {e.stderr}")
            return False
        except json.JSONDecodeError:
            log.error(f"Failed to parse video info for {url}")
            return False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Validation error for {url}: {e}")
            return False

    def _check_video_resolution(self, video_info: dict, url: str) -> bool:
        width = video_info.get('width')
        height = video_info.get('height')

        if width is None or height is None:
            log.warning(f"Could not determine resolution for {url}")
            return False

        min_width, min_height = self.frame_dimensions
        if width < min_width or height < min_height:
            log.warning(f"Resolution {width}x{height} below minimum {min_width}x{min_height}")
            return False

        return True

    @staticmethod
    def _is_valid_duration_format(duration_str: str) -> bool:
        return bool(re.match(r'^(\d+:)?\d+:\d+$|^\d+$', duration_str))

    @staticmethod
    def _convert_duration_to_minutes(duration_str: str) -> float:
        try:
            parts = list(map(int, duration_str.split(':')))
            if len(parts) == 3:
                return parts[0] * 60 + parts[1] + parts[2] / 60
            elif len(parts) == 2:
                return parts[0] + parts[1] / 60
            elif len(parts) == 1:
                return parts[0] / 60
            return 0.0
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _cleanup_temp_dir(temp_dir_path: str) -> None:
        try:
            import shutil
            shutil.rmtree(temp_dir_path)
            log.info(f"Cleaned up temporary directory: {temp_dir_path}")
        except Exception as e:
            log.error(f"Failed to cleanup temporary directory {temp_dir_path}: {e}")

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        patterns = [
            r'rutube\.ru/video/([a-f0-9]{32})',
            r'rutube\.ru/play/embed/([a-f0-9]{32})',
            r'rutube\.ru/video/embed/([a-f0-9]{32})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        log.warning(f"Could not extract video ID from URL: {url}")
        return None


async def main():
    rutube_url = "https://rutube.ru/video/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6/"
    
    downloader = VideoDownloader(max_duration=5)
    
    try:
        temp_video_path = downloader.download_video_to_temp(rutube_url)
        
        if temp_video_path:
            log.info(f"Video downloaded to: {temp_video_path}")
            VideoDownloader._cleanup_temp_dir(os.path.dirname(temp_video_path))
        else:
            log.error("Video download failed.")
    except asyncio.CancelledError:
        log.info("Download was cancelled")


if __name__ == "__main__":
    asyncio.run(main())