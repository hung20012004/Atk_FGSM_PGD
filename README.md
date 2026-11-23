# d:\atk — Tổng hợp mã nguồn tấn công đối kháng (adversarial attack)

Toàn bộ mã trong `repos/` được clone nguyên trạng từ GitHub, **không chỉnh sửa**.

## Cấu trúc thư mục

```
d:\atk
├── README.md
├── weights/                      # trọng số tải từ GitHub release chính thức
│   ├── yolov5s.pt  yolov7.pt  yolov8n.pt  yolo26n.pt
└── repos/
    ├── adversarial-attacks-pytorch/   # torchattacks: PGD, FGSM, C&W
    ├── OSFD/                          # OSFD (AAAI 2024)
    ├── FSA/                           # FSA (PRICAI 2024)
    ├── yolov5/                        # ultralytics/yolov5
    ├── yolov7/                        # WongKinYiu/yolov7
    └── ultralytics/                   # YOLOv8 + YOLO26 (v8.4.216, đã pip install -e)
```

## File core của từng thuật toán

### 1. PGD, FGSM, C&W — `repos/adversarial-attacks-pytorch` (Harry24k/adversarial-attacks-pytorch)

| Thuật toán | File core | Class |
|---|---|---|
| Lớp cơ sở | `torchattacks/attack.py` | `Attack` |
| FGSM | `torchattacks/attacks/fgsm.py` | `FGSM(model, eps=8/255)` |
| PGD (L∞) | `torchattacks/attacks/pgd.py` | `PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)` |
| C&W (L2) | `torchattacks/attacks/cw.py` | `CW(model, c=1, kappa=0, steps=50, lr=0.01)` |

- Đầu vào: ảnh `[0,1]`, shape `(N,C,H,W)`; `model(x)` phải trả về logits `(N, num_classes)` kèm nhãn `(N,)`, tức là thiết kế cho model **phân loại**.

### 2. OSFD — `repos/OSFD` (wakuwu/OSFD, AAAI 2024)
"Transferable Adversarial Attacks for Object Detection using Object-Aware Significant Feature Distortion"

| Vai trò | File |
|---|---|
| **Loss OSFD (core)** | `attack/ours/OSFD.py`, `MSE(k * feat_clean, feat_adv)`, `k=3.0` |
| Trích feature sạch / ghép loss theo stage | `attack/comparing/TransferAttack.py` |
| Bước cập nhật nhiễu (I-FGSM) | `attack/base/IFGSM.py` |
| Momentum | `attack/base/MI.py` |
| Input diversity | `attack/base/DI.py` |
| RRB (xoay/resize theo box + nhiễu Gauss) | `attack/base/RRB.py` |
| Pipeline (tiền xử lý → loss → cập nhật nhiễu) | `attack/utils/pipelines.py` |
| Vòng lặp tấn công + đánh giá mAP | `attack/Attack.py` |
| Entry point | `run_perturb.py config/attack_{yolov3,vfnet,faster_rcnn,swin}.yaml` |
| Tham số mặc định | `config/default.py`, `config/base.yaml` |

- Phụ thuộc **mmdetection 2.28.2 / mmcv-full 1.7.1 / PyTorch 1.12.1 / Python 3.10** (xem README gốc). Không chạy được với môi trường hiện tại (Python 3.13, torch 2.7.1), phải tạo conda env riêng.
- Ảnh trong pipeline ở thang `[0,255]`, `epsilon` mặc định = 5.

### 3. FSA — `repos/FSA` (LMBTough/FSA, PRICAI 2024, arXiv 2408.12670)
"Leveraging Information Consistency in Frequency and Spatial Domain for Adversarial Attacks"
(README trong repo vẫn ghi tên cũ là "FCA")

| Vai trò | File | Class |
|---|---|---|
| Biến đổi DCT/IDCT | `attacks/dct.py` | `dct_2d`, `idct_2d` |
| Lớp cơ sở | `attacks/attack.py` | `Attack` |
| **FSA + I-FGSM** | `attacks/i_fgsm.py` | `DctIFGSM` (bản gốc: `IFGSM`) |
| **FSA + MI-FGSM** | `attacks/mi_fgsm.py` | `DctMIFGSM` |
| **FSA + DI-FGSM** | `attacks/di_fgsm.py` | `DctDIFGSM` |
| **FSA + TI-FGSM** | `attacks/ti_fgsm.py` | `DctTIFGSM` |
| **FSA + PGD** | `attacks/pgd.py` | `DctPGD` |
| Script thí nghiệm ImageNet | `main.py` | |

- Phần FSA nằm trong các class `Dct*`: chỉ cập nhật trước những điểm ảnh có dấu gradient miền không gian trùng với dấu gradient miền tần số (DCT), sau đó mới cập nhật phần còn lại.
- Thiết kế cho model **phân loại ImageNet**: `model = nn.Sequential(Normalize, model, Softmax)`, gọi `attack(input, target)`.
- `__pycache__/` và `.ipynb_checkpoints/` là file thừa có sẵn trong repo.

## Hướng dẫn chạy

Xem `USAGE.md`.

## Model YOLO

| Phiên bản | Repo | Trọng số |
|---|---|---|
| YOLOv5 | `repos/yolov5` | `weights/yolov5s.pt` |
| YOLOv7 | `repos/yolov7` | `weights/yolov7.pt` |
| YOLOv8 | `repos/ultralytics` | `weights/yolov8n.pt` |
| YOLO26 | `repos/ultralytics` | `weights/yolo26n.pt` |

Ghi chú tương thích:
- `yolov5` và `yolov7` đều có package `models/` và `utils/` ở cấp cao nhất, nên không import được cả hai trong cùng một tiến trình Python.
- `yolov7/models/experimental.py` gọi `torch.load` mà không truyền `weights_only=False`. Với torch ≥ 2.6, lệnh này lỗi khi nạp checkpoint `.pt`.
- Các thuật toán trên (trừ OSFD) đều nhận model phân loại; OSFD gắn với detector của mmdetection. Repo nào cũng không hỗ trợ sẵn YOLOv5/v7/v8/YOLO26.
