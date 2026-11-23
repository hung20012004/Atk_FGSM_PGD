import argparse
from atk_core import SUPPORTED_ATTACKS, SUPPORTED_MODELS, evaluate_attack


def _parse_float_expr(val: str) -> float:
    if "/" in val:
        num, den = val.split("/", 1)
        return float(num) / float(den)
    return float(val)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Adversarial Attack Runner for YOLOv5, YOLOv7/YOLO-CCA, YOLOv8, and YOLO26"
    )
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--attack", type=str, required=True, choices=SUPPORTED_ATTACKS)
    parser.add_argument("--source", type=str, required=True, help="Image file or folder of images.")
    parser.add_argument("--labels", type=str, default=None, help="Optional YOLO GT labels directory.")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--eps", type=_parse_float_expr, default=8 / 255)
    parser.add_argument("--alpha", type=_parse_float_expr, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-save-adv", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_attack(
        model_name=args.model,
        attack_name=args.attack,
        source=args.source,
        weights=args.weights,
        labels_dir=args.labels,
        output_dir=args.output,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        k=args.k,
        c=args.c,
        kappa=args.kappa,
        lr=args.lr,
        conf=args.conf,
        iou=args.iou,
        lpips_net=args.lpips_net,
        device=args.device,
        save_adv=not args.no_save_adv,
        save_vis=args.save_vis,
        verbose=True,
    )


if __name__ == "__main__":
    main()
