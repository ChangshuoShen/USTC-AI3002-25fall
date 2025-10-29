# 工具文件
import os
import random
import torch
from tqdm import tqdm
from autoencoder import AE
from pathlib import Path
import numpy as np
from datetime import datetime

def get_date_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    if path:
        os.path.exists(path) or os.makedirs(path, exist_ok=True)

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path) / get_date_str()
    print(f"Results will be saved to '{results_path}'")
    return results_path

def ae_encode(ae: AE, dataset: np.ndarray) -> np.ndarray:
    """
    Use autoencoders to encode data.

    Args:
        - ae(AE): Autoencoder
        - dataset(np.ndarray): shape is (B, C, H, W)
    """
    ae.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae.to(device)
    B, C, H, W = dataset.shape
    img_tensor = torch.from_numpy(dataset)
    img_tensor = img_tensor.to(device).float()

    encoded_list = []
    print("AE encoding ...")
    with torch.no_grad():
        for i in tqdm(range(B)):
            img = img_tensor[i].unsqueeze(0)  # [1, C, H, W]
            img_encoded = ae.encoder(img)
            encoded_list.append(img_encoded.squeeze(0))  # [dim]

    encoded_imgs = torch.stack(encoded_list)  # [B, dim]
    if device == "cuda":
        return encoded_imgs.detach().cpu().numpy()
    return encoded_imgs.numpy()