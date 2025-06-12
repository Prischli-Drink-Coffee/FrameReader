import subprocess
import re
import cv2
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import requests
import logging
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip

from src.utils.custom_logging import setup_logging

log = setup_logging()


class VideoDownloader:
    def __init__(self, 
                 min_duration: int = 0, 
                 max_duration: int = 10, # Max duration in minutes
                 quality: str = 'best',
                 audio_fps: int = 16000,
                 n_frames: int = 100,
                 frame_dimensions: Tuple[int, int] = (224, 224)
                 ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.quality = quality
        self.audio_fps = audio_fps
        self.frame_dimensions = frame_dimensions
        self.n_frames = n_frames

    def download_video_to_temp(self, url: str) -> Optional[str]:
        """
        Downloads a video from the given URL to a temporary file.
        Returns the path to the temporary video file if successful, None otherwise.
        """
        video_id = self._extract_video_id(url)
        if video_id is None:
            log.warning(f"Invalid URL, unable to extract video ID from {url}.")
            return None

        # Check video validity before downloading
        if not self._is_video_valid(url):
            log.warning(f"Video {url} does not meet the criteria. Skipping download.")
            return None

        # Create a temporary directory for the video
        temp_dir = tempfile.mkdtemp(prefix="fr_video_")
        temp_video_path = Path(temp_dir) / f"{video_id}.mp4"

        command = [
            'yt-dlp',
            '-P', str(temp_dir),
            '-f', self.quality,
            '-o', f"{video_id}.%(ext)s",
            url
        ]
        
        try:
            log.info(f"Attempting to download video from {url} to {temp_video_path}")
            # Use subprocess.run with shell=False for better security and explicit command list
            subprocess.run(command, check=True, capture_output=True, text=True)
            log.info(f"Successfully downloaded video from {url} to {temp_video_path}")
            
            # Verify if the file exists and has content
            if not temp_video_path.exists() or temp_video_path.stat().st_size == 0:
                log.error(f"Downloaded video file is empty or does not exist: {temp_video_path}")
                self._cleanup_temp_dir(temp_dir)
                return None

            return str(temp_video_path)
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to download video from {url}. Error: {e.stderr}", exc_info=True)
            self._cleanup_temp_dir(temp_dir)
            return None
        except Exception as e:
            log.error(f"An unexpected error occurred during video download from {url}: {e}", exc_info=True)
            self._cleanup_temp_dir(temp_dir)
            return None

    @staticmethod
    def _cleanup_temp_dir(temp_dir_path: str) -> None:
        """Removes the temporary directory and its contents."""
        try:
            import shutil
            shutil.rmtree(temp_dir_path)
            log.info(f"Cleaned up temporary directory: {temp_dir_path}")
        except Exception as e:
            log.error(f"Failed to cleanup temporary directory {temp_dir_path}: {e}")

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        # Improved regex for RuTube video ID extraction
        match = re.search(r'(?:rutube\.ru\/(?:video\/|play\/embed\/)|rutube\.ru\/video\/embed\/)([a-f0-9]{32})', url)
        if match:
            return match.group(1)
        log.warning(f"Could not extract video ID from URL: {url}")
        return None

    def _is_video_valid(self, url: str) -> bool:
        """
        Checks if the video meets duration and resolution criteria using yt-dlp.
        """
        try:
            # Get duration
            duration_result = subprocess.run(
                ['yt-dlp', '--get-duration', url],
                capture_output=True, text=True, check=True, timeout=30
            )
            duration_str = duration_result.stdout.strip()
            log.debug(f"yt-dlp duration output for {url}: {duration_str}")

            if not re.match(r'(\d+):(\d+)(:\d+)?', duration_str):
                log.warning(f"Invalid duration format from yt-dlp: {duration_str}")
                return False

            duration_minutes = self._convert_duration_to_minutes(duration_str)
            if not (self.min_duration <= duration_minutes <= self.max_duration):
                log.warning(f"Video duration {duration_minutes:.2f} min is not within [{self.min_duration}, {self.max_duration}] min.")
                return False

            # Get resolution (yt-dlp can provide this in JSON format)
            info_result = subprocess.run(
                ['yt-dlp', '--dump-json', '--no-warnings', url],
                capture_output=True, text=True, check=True, timeout=30
            )
            video_info = json.loads(info_result.stdout)
            
            width = video_info.get('width')
            height = video_info.get('height')

            if width is None or height is None:
                log.warning(f"Could not determine video resolution for {url}.")
                return False

            if width < self.frame_dimensions[0] or height < self.frame_dimensions[1]:
                log.warning(f"Video resolution {width}x{height} is below the minimum required {self.frame_dimensions[0]}x{self.frame_dimensions[1]}.")
                return False

            return True

        except subprocess.TimeoutExpired:
            log.error(f"yt-dlp command timed out for {url}.")
            return False
        except subprocess.CalledProcessError as e:
            log.error(f"yt-dlp failed for {url}. Stderr: {e.stderr}", exc_info=True)
            return False
        except json.JSONDecodeError:
            log.error(f"Failed to parse yt-dlp JSON output for {url}.")
            return False
        except Exception as e:
            log.error(f"An unexpected error occurred during video validation for {url}: {e}", exc_info=True)
            return False

    @staticmethod
    def _convert_duration_to_minutes(duration_str: str) -> float:
        parts = list(map(int, duration_str.split(':')))
        if len(parts) == 3: # HH:MM:SS
            return parts[0] * 60 + parts[1] + parts[2] / 60
        elif len(parts) == 2: # MM:SS
            return parts[0] + parts[1] / 60
        elif len(parts) == 1: # SS
            return parts[0] / 60
        return 0.0

# Example usage (for testing purposes, not part of the main app flow)
async def main():
    # This URL is an example, replace with a real RuTube video URL for testing
    # Make sure yt-dlp is installed and accessible in the environment
    rutube_url = "https://rutube.ru/video/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6/" # Replace with a valid URL
    
    downloader = VideoDownloader(max_duration=5) # Max 5 minutes for testing
    temp_video_path = downloader.download_video_to_temp(rutube_url)

    if temp_video_path:
        log.info(f"Video downloaded to: {temp_video_path}")
        # You can now use temp_video_path with cv2.VideoCapture
        # For demonstration, we'll just clean it up
        VideoDownloader._cleanup_temp_dir(os.path.dirname(temp_video_path))
    else:
        log.error("Video download failed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())