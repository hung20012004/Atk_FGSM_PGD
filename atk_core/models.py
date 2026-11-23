import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.ops as ops

ROOT_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT_DIR / "repos"
WEIGHTS_DIR = ROOT_DIR / "weights"

SUPPORTED_MODELS = ("yolov5", "yolocca", "yolov7", "yolov8", "yolo26")

DEFAULT_WEIGHTS: Dict[str, Path] = {
    "yolov5": WEIGHTS_DIR / "yolov5s.pt",
    "yolocca": WEIGHTS_DIR / "yolov7.pt",
    "yolov7": WEIGHTS_DIR / "yolov7.pt",
    "yolov8": WEIGHTS_DIR / "yolov8n.pt",
    "yolo26": WEIGHTS_DIR / "yolo26n.pt",
}


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _patch_torch_load_for_legacy_yolo():
    """Patch torch.load to default weights_only=False for PyTorch >= 2.6 legacy checkpoints."""
    orig_load = torch.load

    def _safe_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    torch.load = _safe_load
    return orig_load


def _load_yolov5(weights: str, device: torch.device):
    repo_path = str(REPOS_DIR / "yolov5")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    orig_load = _patch_torch_load_for_legacy_yolo()
    # Checkpoints trained on Linux/Colab pickle pathlib.PosixPath, which cannot be
    # instantiated on Windows -> map it to WindowsPath while unpickling.
    import pathlib

    orig_posix = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        from models.experimental import attempt_load

        net = attempt_load(weights, device=device)
    finally:
        torch.load = orig_load
        pathlib.PosixPath = orig_posix
    names = getattr(net, "names", None)
    return net, names


def _patch_yolov7_detect_out_of_place(net: nn.Module) -> None:
    """Replace in-place slice assignments in YOLOv7 Detect.forward so autograd succeeds."""
    import types

    detect = net.model[-1]
    if not hasattr(detect, "nl") or not hasattr(detect, "m"):
        return

    def _safe_forward(self_det, x):
        z = []
        x = list(x)
        for i in range(self_det.nl):
            xi = self_det.m[i](x[i])
            bs, _, ny, nx = xi.shape
            xi = xi.view(bs, self_det.na, self_det.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i] = xi
            if not self_det.training:
                if self_det.grid[i].shape[2:4] != xi.shape[2:4]:
                    self_det.grid[i] = self_det._make_grid(nx, ny).to(xi.device)
                y = xi.sigmoid()
                xy_raw, wh_raw, conf = y.split((2, 2, self_det.no - 4), dim=4)
                xy = (xy_raw * 2.0 - 0.5 + self_det.grid[i]) * self_det.stride[i]
                wh = (wh_raw * 2.0) ** 2 * self_det.anchor_grid[i]
                y_out = torch.cat((xy, wh, conf), dim=4)
                z.append(y_out.view(bs, -1, self_det.no))
        if self_det.training:
            return x
        if getattr(self_det, "end2end", False) or getattr(self_det, "concat", False):
            return torch.cat(z, 1)
        return (torch.cat(z, 1), x)

    detect.forward = types.MethodType(_safe_forward, detect)


def _load_yolocca(weights: str, device: torch.device):
    repo_path = str(REPOS_DIR / "yolov7")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    orig_load = _patch_torch_load_for_legacy_yolo()
    try:
        from models.experimental import attempt_load

        net = attempt_load(weights, map_location=device)
    finally:
        torch.load = orig_load
    names = getattr(net, "names", None)
    return net, names


def _load_ultralytics(weights: str, device: torch.device):
    from ultralytics import YOLO

    yolo = YOLO(weights)
    net = yolo.model.to(device)
    names = getattr(yolo, "names", getattr(net, "names", None))
    return net, names


class UnifiedYOLO(nn.Module):
    """Unified differentiable wrapper for YOLOv5, YOLOv7/YOLO-CCA, YOLOv8, and YOLO26."""

    def __init__(
        self,
        model_type: str,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        top_k: int = 32,
    ) -> None:
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_type}'. Choose from: {SUPPORTED_MODELS}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_obj = torch.device(device)

        if weights is None:
            weights_path = DEFAULT_WEIGHTS[self.model_type]
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Default weights not found at {weights_path}. Pass --weights explicitly."
                )
            weights = str(weights_path)
        self.weights_path = weights

        if self.model_type == "yolov5":
            net, names = _load_yolov5(weights, self.device_obj)
            self.has_objectness = True
        elif self.model_type in ("yolocca", "yolov7"):
            net, names = _load_yolocca(weights, self.device_obj)
            self.has_objectness = True
        else:
            net, names = _load_ultralytics(weights, self.device_obj)
            self.has_objectness = False

        self.net = net.to(self.device_obj).eval()
        for m in self.net.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        if self.model_type in ("yolocca", "yolov7"):
            _patch_yolov7_detect_out_of_place(self.net)
        for p in self.net.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        # Keep a dummy buffer/param on device so next(model.parameters()).device works
        self._device_anchor = nn.Parameter(
            torch.zeros(1, device=self.device_obj), requires_grad=False
        )

        self.names = names or {}
        self.top_k = top_k

    def get_detection_tensors(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run raw forward pass and return (boxes_xywh, cls_scores, obj_conf)."""
        if self.net.training:
            self.net.eval()
        out = self.net(x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(pred, dict):
            pred = pred.get("one2one", pred.get("one2many", next(iter(pred.values()))))
            if isinstance(pred, (tuple, list)):
                pred = pred[0]

        if self.has_objectness:
            # YOLOv5 / YOLOv7 / YOLO-CCA: (N, A, 5 + nc)
            boxes = pred[..., :4]
            obj_conf = pred[..., 4]
            cls_probs = pred[..., 5:]
            cls_scores = cls_probs * obj_conf.unsqueeze(-1)
        else:
            # YOLOv8 / YOLO26: (N, 4 + nc, A) -> (N, A, 4 + nc)
            pred = pred.permute(0, 2, 1)
            boxes = pred[..., :4]
            cls_scores = pred[..., 4:]
            obj_conf = cls_scores.max(dim=-1).values

        return boxes, cls_scores, obj_conf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classification-logit adapter (N, nc + 1) for torchattacks & FSA.
        Index 0..nc-1: class logits; index nc: background logit.
        """
        _, cls_scores, obj_conf = self.get_detection_tensors(x)
        num_anchors = cls_scores.shape[1]
        k = min(self.top_k, num_anchors)

        top_cls = cls_scores.topk(k=k, dim=1).values.mean(dim=1)  # (N, nc)
        bg_prob = 1.0 - obj_conf.topk(k=k, dim=1).values.mean(dim=1, keepdim=True)
        probs = torch.cat([top_cls, bg_prob], dim=-1).clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(probs / (1.0 - probs))
        return logits

    @torch.no_grad()
    def predict_boxes(
        self,
        x: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        """Return list of (M, 6) tensors [x1, y1, x2, y2, conf, cls_id] per image."""
        boxes_xywh, cls_scores, _ = self.get_detection_tensors(x)
        results: List[torch.Tensor] = []

        for b in range(x.shape[0]):
            confs, cls_ids = cls_scores[b].max(dim=-1)
            mask = confs > conf_thres
            if not mask.any():
                results.append(torch.zeros((0, 6), device=x.device))
                continue

            b_xyxy = _xywh2xyxy(boxes_xywh[b][mask])
            b_conf = confs[mask]
            b_cls = cls_ids[mask].float()

            # Class-aware NMS via coordinate offset
            offset = b_cls * 4096.0
            keep = ops.nms(b_xyxy + offset.unsqueeze(-1), b_conf, iou_thres)[:max_det]
            dets = torch.cat(
                [b_xyxy[keep], b_conf[keep].unsqueeze(-1), b_cls[keep].unsqueeze(-1)],
                dim=-1,
            )
            results.append(dets)

        return results

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale neck feature maps (inputs to final Detect head)."""
        captured: List[torch.Tensor] = []

        def _collect_tensors(obj):
            if isinstance(obj, torch.Tensor):
                captured.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect_tensors(item)

        detect_layer = self.net.model[-1]
        handle = detect_layer.register_forward_pre_hook(
            lambda _mod, inputs: _collect_tensors(inputs)
        )
        try:
            self.net(x)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Failed to capture feature maps from Detect head.")
        return captured
