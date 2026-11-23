import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .attacks import run_attack
from .metrics import LPIPSMeter, compute_map_and_asr, compute_ssim, load_yolo_gt
from .models import DEFAULT_WEIGHTS, REPOS_DIR, UnifiedYOLO
from .utils import collect_images, load_image, save_comparison, save_image


def _reset_yolo_namespace() -> None:
    """Clear top-level 'models' and 'utils' modules so yolov5 and yolov7/yolocca don't collide."""
    for mod_name in list(sys.modules.keys()):
        if mod_name == "models" or mod_name.startswith("models.") or mod_name == "utils" or mod_name.startswith("utils."):
            del sys.modules[mod_name]
    for repo_name in ("yolov5", "yolov7"):
        rp = str(REPOS_DIR / repo_name)
        while rp in sys.path:
            sys.path.remove(rp)


def _auto_labels_dir(source_path: Path, explicit_labels: Optional[str]) -> Optional[Path]:
    if explicit_labels:
        p = Path(explicit_labels)
        return p if p.exists() else None
    base = source_path if source_path.is_dir() else source_path.parent
    candidate = base.parent / "labels"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _build_gt_for_batch(
    batch_paths: List[Path],
    clean_dets_batch: List[torch.Tensor],
    labels_dir: Optional[Path],
    imgsz: int,
) -> Tuple[List[torch.Tensor], bool]:
    """Return (gt_list, using_real_gt). Falls back to clean detections if no GT files exist."""
    if labels_dir is not None:
        gts = [load_yolo_gt(labels_dir / f"{p.stem}.txt", imgsz) for p in batch_paths]
        if any(len(g) > 0 for g in gts):
            return gts, True
    # Clean-prediction pseudo-GT: [x1, y1, x2, y2, cls_id]
    pseudo_gts = []
    for dets in clean_dets_batch:
        if len(dets) == 0:
            pseudo_gts.append(torch.zeros((0, 5), dtype=torch.float32))
        else:
            pseudo_gts.append(torch.cat([dets[:, :4], dets[:, 5:6]], dim=-1).cpu())
    return pseudo_gts, False


def evaluate_attack(
    model_name: str,
    attack_name: str,
    source: str,
    weights: Optional[str] = None,
    labels_dir: Optional[str] = None,
    output_dir: str = "outputs",
    batch_size: int = 4,
    imgsz: int = 640,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    k: float = 3.0,
    c: float = 1.0,
    kappa: float = 0.0,
    lr: float = 0.01,
    conf: float = 0.25,
    iou: float = 0.45,
    lpips_net: str = "alex",
    device: Optional[str] = None,
    save_adv: bool = True,
    save_vis: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a batched adversarial attack and compute LPIPS, SSIM, Success Rate, mAP, and delta mAP."""
    _reset_yolo_namespace()
    resolved_weights = weights or str(DEFAULT_WEIGHTS[model_name])
    if verbose:
        print(f"\n[INFO] Loading model={model_name} weights={resolved_weights}")

    model = UnifiedYOLO(model_name, weights=resolved_weights, device=device)
    lpips_meter = LPIPSMeter(net=lpips_net, device=model.device_obj)

    img_paths = collect_images(source)
    gt_dir = _auto_labels_dir(Path(source), labels_dir)
    out_path = Path(output_dir) / f"{model_name}_{attack_name}"
    if save_adv or save_vis:
        out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"[INFO] Attack={attack_name} | images={len(img_paths)} | batch_size={batch_size} "
            f"| eps={eps:.5f} | alpha={alpha:.5f} | steps={steps} | device={model.device_obj}"
        )

    all_clean_dets: List[torch.Tensor] = []
    all_adv_dets: List[torch.Tensor] = []
    all_gts: List[torch.Tensor] = []
    used_real_gt = False

    lpips_vals: List[float] = []
    ssim_vals: List[float] = []
    linf_vals: List[float] = []
    l2_vals: List[float] = []
    img_success_count = 0
    total_nonempty_imgs = 0

    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start : start + batch_size]
        clean_batch = torch.cat([load_image(p, imgsz=imgsz) for p in batch_paths], dim=0).to(model.device_obj)

        # Low-conf predictions for mAP precision-recall curve; conf-filtered for ASR & vis
        clean_dets_eval = model.predict_boxes(clean_batch, conf_thres=0.001, iou_thres=iou)
        clean_dets_vis = model.predict_boxes(clean_batch, conf_thres=conf, iou_thres=iou)

        adv_batch = run_attack(
            model,
            clean_batch,
            attack_name=attack_name,
            eps=eps,
            alpha=alpha,
            steps=steps,
            k=k,
            c=c,
            kappa=kappa,
            lr=lr,
        )

        adv_dets_eval = model.predict_boxes(adv_batch, conf_thres=0.001, iou_thres=iou)
        adv_dets_vis = model.predict_boxes(adv_batch, conf_thres=conf, iou_thres=iou)

        batch_gts, is_real = _build_gt_for_batch(batch_paths, clean_dets_vis, gt_dir, imgsz)
        used_real_gt = used_real_gt or is_real

        batch_lpips = lpips_meter(clean_batch, adv_batch)
        batch_ssim = compute_ssim(clean_batch, adv_batch)
        diff = (adv_batch - clean_batch).abs()
        batch_linf = diff.flatten(1).max(dim=1).values
        batch_l2 = torch.norm(diff.flatten(1), dim=1)

        lpips_vals.extend(batch_lpips.cpu().tolist())
        ssim_vals.extend(batch_ssim.cpu().tolist())
        linf_vals.extend(batch_linf.cpu().tolist())
        l2_vals.extend(batch_l2.cpu().tolist())

        all_clean_dets.extend([d.cpu() for d in clean_dets_eval])
        all_adv_dets.extend([d.cpu() for d in adv_dets_eval])
        all_gts.extend(batch_gts)

        for idx, p in enumerate(batch_paths):
            c_det = clean_dets_vis[idx]
            a_det = adv_dets_vis[idx]
            if len(c_det) > 0:
                total_nonempty_imgs += 1
                if len(a_det) < len(c_det):
                    img_success_count += 1

            if save_adv:
                save_image(adv_batch[idx : idx + 1], out_path / f"{p.stem}_adv.png")
            if save_vis:
                save_comparison(
                    clean_batch[idx : idx + 1],
                    adv_batch[idx : idx + 1],
                    c_det,
                    a_det,
                    model.names,
                    out_path / f"{p.stem}_vis.png",
                )

            if verbose:
                c_max = c_det[:, 4].max().item() if len(c_det) else 0.0
                a_max = a_det[:, 4].max().item() if len(a_det) else 0.0
                print(
                    f"  [{p.name}] det: {len(c_det)}->{len(a_det)} "
                    f"(max_conf {c_max:.3f}->{a_max:.3f}) | "
                    f"LPIPS={batch_lpips[idx].item():.4f} SSIM={batch_ssim[idx].item():.4f} "
                    f"Linf={batch_linf[idx].item():.5f}"
                )

    clean_metrics = compute_map_and_asr(all_clean_dets, all_gts)
    # For box-level ASR we filter adv detections by conf threshold so suppressed low-conf boxes count as evaded
    adv_dets_conf = [d[d[:, 4] >= conf] if len(d) else d for d in all_adv_dets]
    adv_metrics_conf = compute_map_and_asr(adv_dets_conf, all_gts)
    adv_metrics_eval = compute_map_and_asr(all_adv_dets, all_gts)

    clean_map50 = clean_metrics["mAP50"]
    clean_map50_95 = clean_metrics["mAP50_95"]
    adv_map50 = adv_metrics_eval["mAP50"]
    adv_map50_95 = adv_metrics_eval["mAP50_95"]
    delta_map50 = clean_map50 - adv_map50
    delta_map50_95 = clean_map50_95 - adv_map50_95

    mean_lpips = sum(lpips_vals) / max(len(lpips_vals), 1)
    mean_ssim = sum(ssim_vals) / max(len(ssim_vals), 1)
    mean_linf = sum(linf_vals) / max(len(linf_vals), 1)
    mean_l2 = sum(l2_vals) / max(len(l2_vals), 1)
    asr_box = adv_metrics_conf["asr_box"]
    asr_img = (100.0 * img_success_count / total_nonempty_imgs) if total_nonempty_imgs > 0 else 0.0

    summary = {
        "model": model_name,
        "attack": attack_name,
        "num_images": len(img_paths),
        "gt_mode": "yolo_gt" if used_real_gt else "clean_ref",
        "lpips": round(mean_lpips, 5),
        "ssim": round(mean_ssim, 5),
        "asr_box_pct": round(asr_box, 2),
        "asr_img_pct": round(asr_img, 2),
        "clean_mAP50": round(clean_map50, 4),
        "adv_mAP50": round(adv_map50, 4),
        "delta_mAP50": round(delta_map50, 4),
        "clean_mAP50_95": round(clean_map50_95, 4),
        "adv_mAP50_95": round(adv_map50_95, 4),
        "delta_mAP50_95": round(delta_map50_95, 4),
        "linf": round(mean_linf, 5),
        "l2": round(mean_l2, 3),
        "output_dir": str(out_path),
    }

    if verbose:
        print(
            f"[RESULT] {model_name} + {attack_name} ({summary['gt_mode']}): "
            f"LPIPS={summary['lpips']:.4f} | SSIM={summary['ssim']:.4f} | "
            f"ASR(box)={summary['asr_box_pct']:.2f}% | "
            f"mAP@0.5: {summary['clean_mAP50']:.4f} -> {summary['adv_mAP50']:.4f} "
            f"(d_mAP={summary['delta_mAP50']:+.4f}) | "
            f"mAP@0.5:0.95: {summary['clean_mAP50_95']:.4f} -> {summary['adv_mAP50_95']:.4f} "
            f"(d_mAP={summary['delta_mAP50_95']:+.4f})"
        )
    return summary
