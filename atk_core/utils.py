from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image, ImageDraw
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: Union[str, Path]) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise FileNotFoundError(f"No images found in directory: {p}")
        return files
    raise FileNotFoundError(f"Source path not found: {p}")


def load_image(path: Union[str, Path], imgsz: int = 640) -> torch.Tensor:
    """Load an image as float32 tensor (1, 3, imgsz, imgsz) in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_u8)


def save_image(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def _draw_dets(img: Image.Image, dets: torch.Tensor, names: Dict[int, str], color: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for row in dets.cpu().tolist():
        x1, y1, x2, y2, conf, cls_id = row
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_int = int(cls_id)
        label = names.get(cls_int, str(cls_int)) if isinstance(names, dict) else (
            names[cls_int] if isinstance(names, (list, tuple)) and cls_int < len(names) else str(cls_int)
        )
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{label} {conf:.2f}", fill=color)
    return out


def save_comparison(
    clean_t: torch.Tensor,
    adv_t: torch.Tensor,
    clean_dets: torch.Tensor,
    adv_dets: torch.Tensor,
    names,
    path: Union[str, Path],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean_img = _draw_dets(tensor_to_pil(clean_t), clean_dets, names, "lime")
    adv_img = _draw_dets(tensor_to_pil(adv_t), adv_dets, names, "red")
    w, h = clean_img.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(clean_img, (0, 0))
    canvas.paste(adv_img, (w, 0))
    canvas.save(path)
