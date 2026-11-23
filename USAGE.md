# Hướng dẫn chạy từng repo (theo hướng dẫn gốc của tác giả)

Mỗi repo được chạy riêng, đúng như tác giả thiết kế. Danh sách file core của từng thuật toán: xem `README.md`.

## 1. OSFD (`repos/OSFD`, cần môi trường riêng)

### Cài môi trường
```shell
conda create -n OSFD python=3.10 -y
conda activate OSFD
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install "mmcv-full==1.7.1"
mim install "mmdet==2.28.2"
pip install imgaug tensorboard
```

### Chuẩn bị dữ liệu (VOC2012 → COCO, 2000 ảnh)
```shell
cd repos/OSFD/data
# tải http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar rồi giải nén tại đây
cd ..
python tools/datasets/split_dataset.py --num 2000 --voc_type "VOC2012" --paper True
python tools/datasets/voc12_to_coco.py --num 2000 --img_size 800 --voc_type "VOC2012"
```

### Tải checkpoint
```shell
cd ummdet/checkpoints/models/
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest .
mim download mmdet --config vfnet_r50_fpn_mstrain_2x_coco --dest .
mim download mmdet --config faster_rcnn_r101_caffe_fpn_mstrain_3x_coco --dest .
mim download mmdet --config mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco --dest .
# (các model đánh giá khác: yolof, yolox, fcos, detr — xem README gốc)
```

### Chạy
Sửa `project_path` trong `config/attack_{model_name}.yaml` thành đường dẫn tuyệt đối tới `repos/OSFD/`, rồi chạy:
```shell
cd repos/OSFD
python run_perturb.py config/attack_yolov3.yaml   # hoặc attack_vfnet / attack_faster_rcnn / attack_swin
```
Tham số chính trong yaml: `steps`, `max_epoch`, `epsilon` (thang 0–255), `base_attack` (`MI`, `DI`, `RRB`), `OSFD.k`.

## 2. FSA (`repos/FSA`, model phân loại ImageNet)

Ví dụ đầy đủ nằm trong `repos/FSA/main.py`. Cách dùng theo README của repo:
```python
import sys; sys.path.insert(0, r"d:\atk\repos\FSA")
import torch.nn as nn
import torchvision.transforms as T
from attacks.i_fgsm import DctIFGSM          # DctMIFGSM / DctDIFGSM / DctTIFGSM / DctPGD

model = nn.Sequential(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      model, nn.Softmax(dim=-1)).to(device)
atk = DctIFGSM(model, eps=eps, alpha=alpha, steps=steps)
adv = atk(images, labels)    # images: (N,C,H,W), labels: (N,)
```

## 3. PGD / FGSM / C&W (`repos/adversarial-attacks-pytorch`, model phân loại)

```python
import sys; sys.path.insert(0, r"d:\atk\repos\adversarial-attacks-pytorch")
import torchattacks

atk = torchattacks.FGSM(model, eps=8/255)
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
atk = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
adv = atk(images, labels)    # images ở [0,1]
```

## 4. Chạy nhận diện YOLO (inference thông thường)

```shell
# YOLOv5
cd repos/yolov5
python detect.py --weights ../../weights/yolov5s.pt --source <ảnh hoặc thư mục>

# YOLOv7 / YOLO-CCA (trỏ --weights tới file .pt của bạn)
cd repos/yolov7
python detect.py --weights ../../weights/yolov7.pt --source <ảnh hoặc thư mục>

# YOLOv8 / YOLO26
yolo predict model=weights/yolov8n.pt source=<ảnh hoặc thư mục>
yolo predict model=weights/yolo26n.pt source=<ảnh hoặc thư mục>
```

Lưu ý: với torch ≥ 2.6, `repos/yolov7` có thể lỗi khi nạp `.pt` vì `torch.load` mặc định `weights_only=True`.
