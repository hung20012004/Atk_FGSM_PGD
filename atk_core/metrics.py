from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.ops as ops


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w_2d = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    return w_2d.expand(channels, 1, window_size, window_size).contiguous()


@torch.no_grad()
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Compute per-image SSIM for (B, C, H, W) tensors in [0, 1]. Returns shape (B,)."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    channels = img1.shape[1]
    window = _gaussian_window(window_size, sigma, channels, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den.clamp(min=1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


class LPIPSMeter:
    """Lazy-loaded LPIPS evaluator for (B, 3, H, W) tensors in [0, 1]."""

    def __init__(self, net: str = "alex", device: Optional[torch.device] = None) -> None:
        self.net_name = net
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fn = None

    def _ensure_loaded(self):
        if self._fn is None:
            import lpips

            self._fn = lpips.LPIPS(net=self.net_name, verbose=False).to(self.device).eval()
            for p in self._fn.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        x1 = img1.to(self.device) * 2.0 - 1.0
        x2 = img2.to(self.device) * 2.0 - 1.0
        scores = self._fn(x1, x2)
        return scores.view(-1)


def load_yolo_gt(label_path: Path, imgsz: int) -> torch.Tensor:
    """Load YOLO txt label file (cls cx cy w h normalized) -> (M, 5) [x1, y1, x2, y2, cls_id]."""
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = float(parts[0])
        cx, cy, w, h = (float(v) * imgsz for v in parts[1:5])
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        rows.append([x1, y1, x2, y2, cls_id])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """101-point COCO-style interpolated AP."""
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    x = torch.linspace(0, 1, 101)
    # Find indices where recall >= x
    inds = torch.searchsorted(mrec, x, right=False)
    ap = mpre[inds.clamp(max=mpre.shape[0] - 1)].mean().item()
    return float(ap)


def compute_map_and_asr(
    preds_per_image: List[torch.Tensor],
    gt_per_image: List[torch.Tensor],
    iou_thresholds: Optional[List[float]] = None,
    asr_iou: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP@0.5, mAP@0.5:0.95, and box-level Attack Success Rate (ASR).
    - preds_per_image: list of (M_i, 6) [x1, y1, x2, y2, conf, cls_id]
    - gt_per_image: list of (G_i, 5) [x1, y1, x2, y2, cls_id]
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 .. 0.95

    all_gt_classes = set()
    total_gt = 0
    for gt in gt_per_image:
        if len(gt) > 0:
            total_gt += len(gt)
            all_gt_classes.update(gt[:, 4].long().tolist())

    if total_gt == 0:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "asr_box": 0.0, "total_gt": 0, "missed_gt": 0}

    # Box-level ASR at asr_iou: fraction of GT boxes NOT matched by any prediction of same class
    missed_gt = 0
    for preds, gt in zip(preds_per_image, gt_per_image):
        if len(gt) == 0:
            continue
        if len(preds) == 0:
            missed_gt += len(gt)
            continue
        ious = ops.box_iou(gt[:, :4].cpu(), preds[:, :4].cpu())  # (G, M)
        same_cls = gt[:, 4:5].cpu().long() == preds[:, 5].cpu().long().unsqueeze(0)
        valid_ious = ious * same_cls.float()
        matched = valid_ious.max(dim=1).values >= asr_iou
        missed_gt += int((~matched).sum().item())

    asr_box = 100.0 * missed_gt / total_gt

    # Compute per-class AP across IoU thresholds
    aps_per_thr: List[List[float]] = [[] for _ in iou_thresholds]
    for cls_id in sorted(all_gt_classes):
        cls_preds = []
        cls_gts: Dict[int, torch.Tensor] = {}
        n_gt_cls = 0
        for img_idx, (preds, gt) in enumerate(zip(preds_per_image, gt_per_image)):
            gt_mask = gt[:, 4].long() == cls_id if len(gt) else torch.zeros(0, dtype=torch.bool)
            cls_gts[img_idx] = gt[gt_mask, :4].cpu() if gt_mask.any() else torch.zeros((0, 4))
            n_gt_cls += len(cls_gts[img_idx])

            if len(preds):
                p_mask = preds[:, 5].long() == cls_id
                if p_mask.any():
                    for row in preds[p_mask].cpu():
                        cls_preds.append((float(row[4]), img_idx, row[:4]))

        if n_gt_cls == 0:
            continue
        if not cls_preds:
            for t_idx in range(len(iou_thresholds)):
                aps_per_thr[t_idx].append(0.0)
            continue

        cls_preds.sort(key=lambda x: x[0], reverse=True)
        for t_idx, iou_thr in enumerate(iou_thresholds):
            tp = torch.zeros(len(cls_preds))
            fp = torch.zeros(len(cls_preds))
            used = {img_idx: torch.zeros(len(g), dtype=torch.bool) for img_idx, g in cls_gts.items()}

            for p_idx, (_, img_idx, box) in enumerate(cls_preds):
                g_boxes = cls_gts[img_idx]
                if len(g_boxes) == 0:
                    fp[p_idx] = 1.0
                    continue
                ious = ops.box_iou(box.unsqueeze(0), g_boxes).squeeze(0)
                best_iou, best_g = ious.max(dim=0)
                if best_iou.item() >= iou_thr and not used[img_idx][best_g]:
                    tp[p_idx] = 1.0
                    used[img_idx][best_g] = True
                else:
                    fp[p_idx] = 1.0

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recall = tp_cum / max(n_gt_cls, 1)
            precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-12)
            aps_per_thr[t_idx].append(_compute_ap(recall, precision))

    map_per_thr = [sum(aps) / len(aps) if aps else 0.0 for aps in aps_per_thr]
    return {
        "mAP50": map_per_thr[0],
        "mAP50_95": sum(map_per_thr) / len(map_per_thr),
        "asr_box": asr_box,
        "total_gt": total_gt,
        "missed_gt": missed_gt,
    }
