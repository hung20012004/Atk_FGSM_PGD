import argparse
from attacks.i_fgsm import IFGSM, DctIFGSM
from attacks.di_fgsm import DIFGSM,DctDIFGSM
from attacks.mi_fgsm import MIFGSM,DctMIFGSM
from attacks.ti_fgsm import TIFGSM,DctTIFGSM
from attacks.pgd import PGD,DctPGD
from PIL import Image
import torch
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import *
from tqdm import tqdm
import random
from torch.utils import data
from torchvision import transforms as T


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)

device = "cuda" if torch.cuda.is_available() else "cpu"


args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="resnet50",choices=[
    "densenet121", "inception_v3", "vgg16","mobilenet_v2","googlenet","efficientnet_b0","vgg19","mobilenet_v3_large"])
args.add_argument("--steps", type=int, default=5,choices=[5,10,16])
args.add_argument("--eps", type=float, default=1,choices=[1,2])

args = args.parse_args()

def exp(model,eps=1/255,steps=10):
    alpha = eps / steps
    mi_fgsm = MIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    di_fgsm = DIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    ti_fgsm = TIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    i_fgsm = IFGSM(model, eps=eps, alpha=alpha, steps=steps)
    pgd = PGD(model, eps=eps, alpha=alpha, steps=steps)
    dct_mi_fgsm = DctMIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    dct_di_fgsm = DctDIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    dct_ti_fgsm = DctTIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    dct_i_fgsm = DctIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    dct_pgd = DctPGD(model, eps=eps, alpha=alpha, steps=steps)
    
    mi_fgsm_func = lambda data,label: model(mi_fgsm(data,label)).argmax(dim=-1)
    di_fgsm_func = lambda data,label: model(di_fgsm(data,label)).argmax(dim=-1)
    ti_fgsm_func = lambda data,label: model(ti_fgsm(data,label)).argmax(dim=-1)
    i_fgsm_func = lambda data,label: model(i_fgsm(data,label)).argmax(dim=-1)
    pgd_func = lambda data,label: model(pgd(data,label)).argmax(dim=-1)

    dct_mi_fgsm_func = lambda data,label: model(dct_mi_fgsm(data,label)).argmax(dim=-1)
    dct_di_fgsm_func = lambda data,label: model(dct_di_fgsm(data,label)).argmax(dim=-1)
    dct_ti_fgsm_func = lambda data,label: model(dct_ti_fgsm(data,label)).argmax(dim=-1)
    dct_i_fgsm_func = lambda data,label: model(dct_i_fgsm(data,label)).argmax(dim=-1)
    dct_pgd_func = lambda data,label: model(dct_pgd(data,label)).argmax(dim=-1)

    success_attacked = {
        "mi_fgsm":0,
        "di_fgsm":0,
        "ti_fgsm":0,
        "i_fgsm":0,
        "dct_mi_fgsm":0,
        "dct_di_fgsm":0,
        "dct_ti_fgsm":0,
        "dct_i_fgsm":0,
        "pgd":0,
        "dct_pgd":0,
        "total":0
    }
    
    batch_size = 16
    pbar = tqdm(total=len(img_batch) // batch_size + 1)
    total = 0
    for i in range(0, len(img_batch), batch_size):
        data = img_batch[i:i + batch_size].to(device)
        label = target_batch[i:i + batch_size].to(device)
        correct_index = model(data).argmax(-1) == label
        data = data[correct_index]
        label = label[correct_index]
        success_attacked["mi_fgsm"] += (mi_fgsm_func(data,label) != label).sum().item()
        success_attacked["di_fgsm"] += (di_fgsm_func(data,label) != label).sum().item()
        success_attacked["ti_fgsm"] += (ti_fgsm_func(data,label) != label).sum().item()
        success_attacked["i_fgsm"] += (i_fgsm_func(data,label) != label).sum().item()
        success_attacked["dct_mi_fgsm"] += (dct_mi_fgsm_func(data,label) != label).sum().item()
        success_attacked["dct_di_fgsm"] += (dct_di_fgsm_func(data,label) != label).sum().item()
        success_attacked["dct_ti_fgsm"] += (dct_ti_fgsm_func(data,label) != label).sum().item()
        success_attacked["dct_i_fgsm"] += (dct_i_fgsm_func(data,label) != label).sum().item()
        success_attacked["pgd"] += (pgd_func(data,label) != label).sum().item()
        success_attacked["dct_pgd"] += (dct_pgd_func(data,label) != label).sum().item()
        success_attacked["total"] += len(data)
        total += len(data)
        
        pbar.set_description(
            f"total:{total}, mi_fgsm:{success_attacked['mi_fgsm'] / total:.4f}, dct_mi_fgsm:{success_attacked['dct_mi_fgsm'] / total:.4f} \
            di_fgsm:{success_attacked['di_fgsm'] / total:.4f}, dct_di_fgsm:{success_attacked['dct_di_fgsm'] / total:.4f} \
            ti_fgsm:{success_attacked['ti_fgsm'] / total:.4f}, dct_ti_fgsm:{success_attacked['dct_ti_fgsm'] / total:.4f} \
            i_fgsm:{success_attacked['i_fgsm'] / total:.4f}, dct_i_fgsm:{success_attacked['dct_i_fgsm'] / total:.4f}"
            f"total:{total}, pgd:{success_attacked['pgd'] / total:.4f}, dct_pgd:{success_attacked['dct_pgd'] / total:.4f}"
        )
        pbar.update(1)
    pbar.close()
    np.savez(f"results/{args.model}_{args.steps}_{args.eps}_pgd.npz", **success_attacked)

if __name__ == "__main__":
    img_batch = torch.load("imagenet_x_batch.pt") # shape: (1000, 3, 224, 224)
    target_batch = torch.load("imagenet_y_batch.pt") # shape: (1000,)
    if "inc" in args.model:
        rs = T.Resize((299,299))
        img_batch = torch.stack([rs(img) for img in img_batch])
        model = eval(args.model)(pretrained=True,aux_logits=False).to(device)
    else:
        model = eval(args.model)(pretrained=True).to(device)
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm_layer = T.Normalize(mean, std)
    sm = nn.Softmax(dim=-1)
    model = nn.Sequential(norm_layer, model, sm).to(device)
    exp(model,eps=args.eps/255,steps=args.steps)