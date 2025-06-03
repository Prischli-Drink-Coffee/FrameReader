import os
import sys
import requests
import logging
import json

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


if __name__ == '__main__':

    test_image_path = "./docs/test.jpg"
    if os.path.exists(test_image_path):
        logger.info(f"Preparing to test with image: {test_image_path}")
        multipart_files = [("images", (os.path.basename(test_image_path), open(test_image_path, "rb"), "image/jpeg"))]
        
        # Test YOLO
        try:
            logger.info("Testing YOLO endpoint...")
            yolo_response = requests.post("http://localhost:8000/generate/yolo", files=multipart_files)
            logger.info(f"YOLO Response Status: {yolo_response.status_code}")
            if yolo_response.status_code == 200:
                logger.info(f"YOLO Response Body: {json.dumps(yolo_response.json(), indent=2)}")
            else:
                logger.error(f"YOLO Response Error Body: {yolo_response.text}")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Could not connect to FastAPI app for YOLO test: {ce}. Is Ray Serve running on http://localhost:8000?")
        except Exception as e:
            logger.error(f"An error occurred during YOLO test: {e}", exc_info=True)
        
        multipart_files_donut = [("images", (os.path.basename(test_image_path), open(test_image_path, "rb"), "image/jpeg"))]

        # Test Donut
        try:
            logger.info("\nTesting Donut endpoint...")
            donut_response = requests.post("http://localhost:8000/generate/donut", files=multipart_files_donut)
            logger.info(f"Donut Response Status: {donut_response.status_code}")
            if donut_response.status_code == 200:
                logger.info(f"Donut Response Body: {json.dumps(donut_response.json(), indent=2)}")
            else:
                logger.error(f"Donut Response Error Body: {donut_response.text}")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Could not connect to FastAPI app for Donut test: {ce}. Is Ray Serve running on http://localhost:8000?")
        except Exception as e:
            logger.error(f"An error occurred during Donut test: {e}", exc_info=True)
    else:
        logger.warning(f"Test image not found at: {test_image_path}. Skipping tests.")
