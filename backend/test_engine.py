import torch
from pathlib import Path
from PIL import Image
import numpy as np
from donut.engine import TRTInferenceEngine


if __name__ == '__main__':
    engine = TRTInferenceEngine(
        model_path='/home/student/projects/TritonServer/models/donut/1/donut_fp16.pt',
        processor_path='/home/student/projects/TritonServer/models/donut/1/checkpoint',
        device=torch.device('cuda')
    )

    # Передача пути
    engine.process_batch([Path('/home/student/projects/TritonServer/docs/test.jpg')])

    # Передача PIL
    img = Image.open(Path('/home/student/projects/TritonServer/docs/test.jpg'))
    imgs = [img]
    engine.process_batch([imgs])

    # Передача тензора
    img = Image.open(Path('/home/student/projects/TritonServer/docs/test.jpg')).convert('RGB')
    img_np = np.array(img)
    image_torch = torch.from_numpy(img_np).to('cuda')
    if len(image_torch.shape) == 3:
        image_torch = image_torch.permute(2, 0, 1)
    engine.process_batch([image_torch])
