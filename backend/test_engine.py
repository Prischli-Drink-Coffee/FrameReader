import torch
from pathlib import Path
from PIL import Image
import numpy as np
from donut.engine import DonutInferenceTRT
from transformers import DonutProcessor
import io


if __name__ == '__main__':

    model_path = Path('/home/student/projects/TritonServer/models/donut/1/donut')
    tensorrt_dir = model_path / 'engine'
    image_path = Path('/home/student/projects/TritonServer/docs/test.jpg')
    image_size = (384, 384)
    batch_size = 1

    with DonutInferenceTRT(
        tensorrt_dir=tensorrt_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=batch_size
    ) as engine:

        processor = DonutProcessor.from_pretrained(model_path, use_fast=True)
        processor.image_processor.size = image_size[::-1]
        processor.image_processor.do_align_long_axis = False
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.thumbnail(image_size[::-1], Image.LANCZOS)

        pixel_values = processor(
            image, 
            return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.squeeze().unsqueeze(0)

        predictions = engine.predict_batch(pixel_values)
        
        print(f"Predictions: {predictions}")

