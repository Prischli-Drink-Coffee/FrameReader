import argparse
import subprocess
import time
import os
from multiprocessing import Process

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tqdm import tqdm
from tritonclient.utils import *


def client_yolo(request_count, batch_size, save_image, index, image_path=None):
    client = httpclient.InferenceServerClient(url="localhost:8000")
    latencies = []
    start = time.time()
    
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
    else:
        image = Image.new('RGB', (512, 512), color=(128, 128, 128))
    
    image_array = np.array(image)
    
    for i in tqdm(range(request_count), position=index):
        batch_data = np.array([image_array] * batch_size)
        
        input_image = httpclient.InferInput(
            "image", batch_data.shape, np_to_triton_dtype(batch_data.dtype)
        )
        input_image.set_data_from_numpy(batch_data)
        
        output_result = httpclient.InferRequestedOutput("result")
        
        request_start = time.time()
        query_response = client.infer(
            model_name="yolo", inputs=[input_image], outputs=[output_result]
        )
        latencies.append(time.time() - request_start)
        
        result = query_response.as_numpy("result")
        
        if save_image and isinstance(result, np.ndarray):
            try:
                import json
                import cv2
                import random
                
                result_obj = json.loads(result[0])
                img_for_drawing = image_array.copy()
                
                if isinstance(result_obj, list) and len(result_obj) > 0:
                    for detection in result_obj:
                        if "boxes" in detection and len(detection["boxes"]) > 0:
                            for j, box in enumerate(detection["boxes"]):
                                x1, y1, x2, y2 = [int(coord) for coord in box]
                                conf = detection["confidence"][j] if "confidence" in detection and j < len(detection["confidence"]) else 0.0
                                cls_id = int(detection["classes"][j]) if "classes" in detection and j < len(detection["classes"]) else 0
                                
                                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                                cv2.rectangle(img_for_drawing, (x1, y1), (x2, y2), color, 2)
                                
                                cls_name = detection.get("class_names", [f"class_{cls_id}"])[j] if j < len(detection.get("class_names", [])) else f"class_{cls_id}"
                                cv2.putText(img_for_drawing, f"{cls_name}: {conf:.2f}", (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imwrite(f"client_{index}_detected_image_{i}.jpg", img_for_drawing)
            except Exception as e:
                print(f"Error saving annotated image: {e}")
    
    print(
        f"Client YOLO: {index} Throughput: {request_count/(time.time()-start)} Avg. Latency: {np.mean(latencies)}"
    )


def client_donut(request_count, batch_size, save_image, index, image_path=None):
    client = httpclient.InferenceServerClient(url="localhost:8000")
    latencies = []
    start = time.time()
    
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
    else:
        image = Image.new('RGB', (384, 384), color=(128, 128, 128))
    
    image_array = np.array(image)
    
    for i in tqdm(range(request_count), position=index):
        batch_data = np.array([image_array] * batch_size)
        
        input_image = httpclient.InferInput(
            "image", batch_data.shape, np_to_triton_dtype(batch_data.dtype)
        )
        input_image.set_data_from_numpy(batch_data)
        
        output_text = httpclient.InferRequestedOutput("text_sequence")
        
        request_start = time.time()
        query_response = client.infer(
            model_name="donut", inputs=[input_image], outputs=[output_text]
        )
        latencies.append(time.time() - request_start)
        
        result = query_response.as_numpy("text_sequence")
        
        if save_image and isinstance(result, np.ndarray):
            try:
                import cv2
                
                img_for_drawing = image_array.copy()
                
                if len(result) > 0:
                    text = str(result[0])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    color = (0, 0, 255)
                    thickness = 1
                    
                    max_width = 80
                    lines = []
                    for j in range(0, len(text), max_width):
                        lines.append(text[j:j+max_width])
                    
                    y = 30
                    for line in lines:
                        cv2.putText(img_for_drawing, line, (30, y), font, font_scale, color, thickness)
                        y += 30
                
                cv2.imwrite(f"client_{index}_donut_image_{i}.jpg", img_for_drawing)
            except Exception as e:
                print(f"Error saving text image: {e}")
    
    print(
        f"Client Donut: {index} Throughput: {request_count/(time.time()-start)} Avg. Latency: {np.mean(latencies)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TritonServer Client for YOLO and Donut models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Number of concurrent clients. Each client sends --requests number of requests.",
    )
    parser.add_argument(
        "--requests", type=int, default=1, help="Number of requests to send."
    )
    parser.add_argument(
        "--static-batch-size",
        type=int,
        default=1,
        help="Number of images to send in a single request",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to image file to use for testing",
    )
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="If provided, generated images will be saved as jpeg files",
    )
    parser.add_argument(
        "--launch-nvidia-smi",
        action="store_true",
        help="Launch nvidia smi in daemon mode and log data to nvidia_smi_output.txt",
    )
    parser.add_argument(
        "--model", type=str, default="yolo", choices=["yolo", "donut"], help="model name"
    )
    args = parser.parse_args()
    
    if args.launch_nvidia_smi:
        nvidia_smi_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-f", "nvidia_smi_output.txt"]
        )
        time.sleep(5)
    
    procs = []
    start_time = time.time()
    
    client_func = client_yolo if args.model == "yolo" else client_donut
    
    for i in range(args.clients):
        procs.append(
            Process(
                target=client_func,
                args=(
                    args.requests,
                    args.static_batch_size,
                    args.save_image,
                    i,
                    args.image_path,
                ),
            )
        )
        procs[-1].start()

    for proc in procs:
        proc.join()
    
    end_time = time.time()
    
    if args.launch_nvidia_smi:
        time.sleep(5)
        nvidia_smi_proc.kill()
    
    print(
        f"Model: {args.model} - Throughput: {(args.requests*args.clients)/(end_time-start_time)} Total Time: {end_time-start_time}"
    )