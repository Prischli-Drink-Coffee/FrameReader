import cv2
import numpy as np
import torch
import tempfile
import os
from PIL import Image
from basicsr.archs.basicvsr_plusplus_arch import BasicVSRPlusPlus
from basicsr.archs.nafnet_arch import NAFNet
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torch.hub import load_state_dict_from_url
from torchvision.transforms import ToTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Депиксилизация с Real-ESRGAN (гитхаб kokutoru слишком специфичен, я честно не смог адаптировать)
def depixelate_frame(frame):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    state_dict = load_state_dict_from_url('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth')
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(state_dict, f)
        model_path = f.name
    try:
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = upsampler.enhance(img, outscale=4)

        result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    finally:
        os.unlink(model_path)

    return result

# Улучшение кадра с NAFNet
def enhance_frame_nafnet(frame):
    model = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )
    model.load_state_dict(load_state_dict_from_url('https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view?usp=sharing'))
    model = model.to(device).eval()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    
    result = output.squeeze().permute(1,2,0).cpu().numpy()
    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Медианное усреднение кадров
def median_weightening(frames):
    return np.median(np.stack(frames), axis=0).astype(np.uint8)

# Динамическое взвешенное усреднение кадров
def dynamic_weightening(frames):
    weights = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sharpness = np.sqrt(gx**2 + gy**2).mean()
        weights.append(sharpness)
    
    weights = np.array(weights)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    
    avg_frame = np.zeros_like(frames[0], dtype=np.float32)
    for i, frame in enumerate(frames):
        avg_frame += frame.astype(np.float32) * weights[i]
    
    return avg_frame.astype(np.uint8)

# Увелечение разрешения с BasicVSR++
def super_resolution(frame):
    model = BasicVSRPlusPlus(num_feat=64, num_block=30)
    model.load_state_dict(load_state_dict_from_url('https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth'))
    model = model.to(device).eval()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(Image.fromarray(img)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
    
    result = output.squeeze().permute(1,2,0).cpu().numpy()
    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# Примерный пайплайн обработки (конечно же, тут надо 24 кадра давать, просто наглядно показал процесс)
def process_video(video_path, use_median=True):
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        depixelated = depixelate_frame(frame)
        enhanced = enhance_frame_nafnet(depixelated)
        processed_frames.append(enhanced)

    if use_median:
        averaged = median_weightening(processed_frames)
    else:
        averaged = dynamic_weightening(processed_frames)
    
    final_result = super_resolution(averaged)
    return final_result
