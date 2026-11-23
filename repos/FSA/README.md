# FCA

Implementation of [FCA: Frequency Consistency Attack Method]

## Usage

Complete examples are shown in `main.py`.Here are some sample code.

```python
from attacks.i_fgsm import DctIFGSM

# load model
model = load_model(...) # load your model
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = T.Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)

dct_i_fgsm = DctIFGSM(model, eps=eps, alpha=alpha, steps=steps)


adv_output = dct_i_fgsm(input, target)
```
input and target are torch.Tensor,shape of input is (N,C,H,W),shape of target is (N,),adv_output is torch.Tensor,shape of adv_output is (N,C,H,W)
