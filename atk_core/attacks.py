import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .models import UnifiedYOLO, REPOS_DIR

SUPPORTED_ATTACKS = (
    "fgsm",
    "pgd",
    "cw",
    "fsa_ifgsm",
    "fsa_pgd",
    "fsa_mifgsm",
    "fsa_difgsm",
    "fsa_tifgsm",
    "osfd",
)


def _infer_target_labels(model: UnifiedYOLO, images: torch.Tensor) -> torch.Tensor:
    """Infer dominant detected class per image (or background class if no detections)."""
    with torch.no_grad():
        logits = model(images)
        # Exclude background column when choosing dominant object class if any object is present
        obj_logits = logits[:, :-1]
        max_vals, cls_ids = obj_logits.max(dim=-1)
        bg_vals = logits[:, -1]
        labels = torch.where(max_vals > bg_vals - 2.0, cls_ids, torch.full_like(cls_ids, logits.shape[1] - 1))
    return labels


def _run_torchattacks(
    model: UnifiedYOLO,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_name: str,
    eps: float,
    alpha: float,
    steps: int,
    c: float,
    kappa: float,
    lr: float,
) -> torch.Tensor:
    repo_path = str(REPOS_DIR / "adversarial-attacks-pytorch")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import torchattacks

    if attack_name == "fgsm":
        atk = torchattacks.FGSM(model, eps=eps)
    elif attack_name == "pgd":
        atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif attack_name == "cw":
        atk = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
    else:
        raise ValueError(f"Unknown torchattacks method: {attack_name}")

    return atk(images, labels)


def _run_fsa(
    model: UnifiedYOLO,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_name: str,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    repo_path = str(REPOS_DIR / "FSA")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    if attack_name == "fsa_ifgsm":
        from attacks.i_fgsm import DctIFGSM

        atk = DctIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_name == "fsa_pgd":
        from attacks.pgd import DctPGD

        atk = DctPGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif attack_name == "fsa_mifgsm":
        from attacks.mi_fgsm import DctMIFGSM

        atk = DctMIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_name == "fsa_difgsm":
        from attacks.di_fgsm import DctDIFGSM

        atk = DctDIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_name == "fsa_tifgsm":
        from attacks.ti_fgsm import DctTIFGSM

        atk = DctTIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    else:
        raise ValueError(f"Unknown FSA method: {attack_name}")

    return atk(images, labels)


def _run_osfd(
    model: UnifiedYOLO,
    images: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    k: float,
) -> torch.Tensor:
    """
    OSFD (AAAI 2024): MSE(k * feat_clean, feat_adv) feature distortion attack
    adapted for YOLO multi-scale neck features.
    """
    with torch.no_grad():
        feats_clean = [f.detach() for f in model.extract_features(images)]

    adv = images.clone().detach()
    adv = adv + torch.empty_like(adv).uniform_(-eps * 0.1, eps * 0.1)
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        feats_adv = model.extract_features(adv)
        loss = sum(F.mse_loss(k * fc, fa) for fc, fa in zip(feats_clean, feats_adv))
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, min=-eps, max=eps)
        adv = torch.clamp(images + delta, min=0.0, max=1.0).detach()

    return adv


def run_attack(
    model: UnifiedYOLO,
    images: torch.Tensor,
    attack_name: str,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    k: float = 3.0,
    c: float = 1.0,
    kappa: float = 0.0,
    lr: float = 0.01,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the selected adversarial attack on a UnifiedYOLO model."""
    name = attack_name.lower()
    if name not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"Unsupported attack '{attack_name}'. Choose from: {SUPPORTED_ATTACKS}"
        )

    images = images.to(model.device_obj)
    if name == "osfd":
        return _run_osfd(model, images, eps=eps, alpha=alpha, steps=steps, k=k)

    if labels is None:
        labels = _infer_target_labels(model, images)
    else:
        labels = labels.to(model.device_obj)

    if name in ("fgsm", "pgd", "cw"):
        return _run_torchattacks(
            model, images, labels, name, eps=eps, alpha=alpha, steps=steps, c=c, kappa=kappa, lr=lr
        )
    return _run_fsa(model, images, labels, name, eps=eps, alpha=alpha, steps=steps)
