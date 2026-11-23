"""Clone required third-party repos into repos/ on a fresh machine."""

import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPOS_DIR = ROOT_DIR / "repos"

REQUIRED_REPOS = {
    "adversarial-attacks-pytorch": "https://github.com/Harry24k/adversarial-attacks-pytorch.git",
    "FSA": "https://github.com/LMBTough/FSA.git",
    "yolov5": "https://github.com/ultralytics/yolov5.git",
    "yolov7": "https://github.com/WongKinYiu/yolov7.git",
}


def main():
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REQUIRED_REPOS.items():
        target = REPOS_DIR / name
        if target.exists():
            print(f"[SKIP] {target} already exists")
            continue
        print(f"[CLONE] {url} -> {target}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)

    weights_dir = ROOT_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    print("\n[DONE] Repos ready.")
    print(
        "Note: Place your .pt weights in weights/ (e.g. yolov5s.pt, yolov7.pt, yolov8n.pt, yolo26n.pt)\n"
        "or set custom paths in atk.py / --weights."
    )


if __name__ == "__main__":
    main()
