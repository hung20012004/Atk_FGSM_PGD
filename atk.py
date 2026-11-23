"""
Cau hinh truc tiep tai day va chay: python atk.py
Ho tro chay 1 hoac nhieu model x nhieu thuat toan tan cong tren nhieu anh cung luc.
"""

import csv
import json
from pathlib import Path
from atk_core import evaluate_attack

# ==============================================================================
# CAU HINH THAM SO TAN CONG (CHINH SUA TAI DAY)
# ==============================================================================
CONFIG = {
    # Chon 1 hoac nhieu model: "yolov5", "yolocca", "yolov8", "yolo26"
    "models": ["yolov5", "yolocca", "yolov8", "yolo26"],

    # Duong dan weights tuy chinh (de None se tu dung weights mac dinh trong weights/)
    "weights": {
        # Model fine-tune tren TT100K (45 lop bien bao giao thong)
        "yolov5": "weights/finetuned/yolov5_best.pt",   # yolov5x, train imgsz=1280
        "yolocca": "weights/finetuned/yoloCCA.pt",      # kien truc YOLOv7 (IDetect)
        "yolov8": "weights/finetuned/yolov8_best.pt",   # train imgsz=2048
        "yolo26": "weights/finetuned/yolo26.pt",        # yolo26l, train imgsz=2048
        # Weights COCO goc: weights/yolov5s.pt, yolov7.pt, yolov8n.pt, yolo26n.pt
    },

    # Chon 1 hoac nhieu thuat toan:
    # "fgsm", "pgd", "cw", "fsa_ifgsm", "fsa_pgd", "fsa_mifgsm", "fsa_difgsm", "fsa_tifgsm", "osfd"
    "attacks": ["pgd", "fsa_ifgsm", "osfd"],

    # Duong dan 1 anh hoac thu muc chua nhieu anh
    # 92 anh TT100K val (anh goc .jpg, cung ten voi bo anh trong repo adversarial-attack-yolov8)
    "source": "data/tt100k/images",

    # Thu muc nhan YOLO (.txt) neu co; de None se tu tim ../labels hoac dung clean baseline
    "labels": "data/tt100k/labels",

    # Kich thuoc batch va anh dau vao (model train o 1280/2048, bien bao rat nho)
    "batch_size": 1,
    "imgsz": 1280,

    # Tham so nhieu doi khang (thang [0, 1])
    "eps": 8 / 255,       # ~0.03137
    "alpha": 2 / 255,     # ~0.00784
    "steps": 10,

    # Tham so rieng cho OSFD & C&W
    "k": 3.0,             # OSFD feature scale k
    "c": 1.0,             # C&W trade-off c
    "kappa": 0.0,         # C&W confidence kappa
    "lr": 0.01,           # C&W Adam learning rate

    # Nguong danh gia detection & LPIPS
    "conf": 0.25,
    "iou": 0.45,
    "lpips_net": "alex",  # "alex" | "vgg" | "squeeze"
    "device": None,       # None = tu chon CUDA neu co, hoac "cuda" / "cpu"

    # Luu ket qua
    "output": "outputs",
    "save_adv": True,     # Luu anh doi khang *_adv.png
    "save_vis": True,     # Luu anh so sanh clean vs adv *_vis.png
}


def main():
    models = CONFIG["models"]
    if isinstance(models, str):
        models = [models]
    attacks = CONFIG["attacks"]
    if isinstance(attacks, str):
        attacks = [attacks]

    weights_map = CONFIG.get("weights") or {}
    results = []

    for model_name in models:
        for attack_name in attacks:
            res = evaluate_attack(
                model_name=model_name,
                attack_name=attack_name,
                source=CONFIG["source"],
                weights=weights_map.get(model_name),
                labels_dir=CONFIG.get("labels"),
                output_dir=CONFIG.get("output", "outputs"),
                batch_size=CONFIG.get("batch_size", 4),
                imgsz=CONFIG.get("imgsz", 640),
                eps=CONFIG.get("eps", 8 / 255),
                alpha=CONFIG.get("alpha", 2 / 255),
                steps=CONFIG.get("steps", 10),
                k=CONFIG.get("k", 3.0),
                c=CONFIG.get("c", 1.0),
                kappa=CONFIG.get("kappa", 0.0),
                lr=CONFIG.get("lr", 0.01),
                conf=CONFIG.get("conf", 0.25),
                iou=CONFIG.get("iou", 0.45),
                lpips_net=CONFIG.get("lpips_net", "alex"),
                device=CONFIG.get("device"),
                save_adv=CONFIG.get("save_adv", True),
                save_vis=CONFIG.get("save_vis", False),
                verbose=True,
            )
            results.append(res)

    out_dir = Path(CONFIG.get("output", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 105)
    print(
        f"{'Model':<10} {'Attack':<12} {'LPIPS':>8} {'SSIM':>8} {'ASR(%)':>8} "
        f"{'c_mAP50':>8} {'a_mAP50':>8} {'d_mAP50':>8} {'c_mAP95':>8} {'a_mAP95':>8} {'d_mAP95':>8}"
    )
    print("-" * 105)
    for r in results:
        print(
            f"{r['model']:<10} {r['attack']:<12} {r['lpips']:>8.4f} {r['ssim']:>8.4f} {r['asr_box_pct']:>8.2f} "
            f"{r['clean_mAP50']:>8.4f} {r['adv_mAP50']:>8.4f} {r['delta_mAP50']:>+8.4f} "
            f"{r['clean_mAP50_95']:>8.4f} {r['adv_mAP50_95']:>8.4f} {r['delta_mAP50_95']:>+8.4f}"
        )
    print("=" * 105)
    print(f"[INFO] Saved report to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
