import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import numpy as np
import scipy
import sys
from PIL import Image
from scipy.stats import spearmanr as spr
import os
import gc
import math
import time
from infidelity_utils import get_exp_infid, get_exp_sens
from captum.attr import Saliency, IntegratedGradients, LRP, GuidedBackprop, NoiseTunnel, GuidedGradCam, InputXGradient, DeepLift
from captum.attr import visualization as viz
sys.path.append("../attacks/")
from utils import torch_to_image, load_image,  convert_relu_to_softplus, change_beta_softplus, heatmap_to_image, get_expl
import mister_ed.utils.pytorch_utils as mister_ed_utils
import argparse
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not", default=False)
argparser.add_argument('--output_dir', type=str, default='results/', help='directory to save results to')
argparser.add_argument('--infidelity_pert', type=str, default="Gaussian", help='infidelity perturbation')
argparser.add_argument('--binary_I',type=bool,
                        help="whether to multiply the explanation with input or not", default=False)
args = argparser.parse_args()
if args.infidelity_pert=="Gaussian":
    result_path = args.output_dir+"softplus/"
elif args.infidelity_pert=="Square":
    result_path = args.output_dir+"softplus_square_pert/"
else:
    raise ValueError("wrong perturbation type.")
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
normalizer = mister_ed_utils.DifferentiableNormalize(mean=data_mean,
                                               std=data_std)
trasform_imagenet = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

imagenet_val = torchvision.datasets.ImageNet(root="../notebooks/data/", split="val", transform=trasform_imagenet)

test_loader = torch.utils.data.DataLoader(
        imagenet_val,
        batch_size=256,
        shuffle=True
    )
dataiter = iter(test_loader)
#################################
seed = 72
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#################################
model = torchvision.models.vgg16(pretrained=True).to(device)
model = model.eval()
softplus_model = convert_relu_to_softplus(model, beta=10)
model = torchvision.models.vgg16(pretrained=True).to(device)
model = model.eval()
#################################
t0 = time.time()
all_infds = 0
all_senss = 0
beta_values = np.arange(0.0, 10.1, 0.5)
i=0
for images, labels in test_loader:
    ind = np.random.randint(len(labels))
    x = normalizer.forward(images[ind:ind+1]).to(device)
    idx = labels[ind].to(device)
    infds = []
    senss = []
    for beta in beta_values:
        softplus_model = change_beta_softplus(softplus_model, beta=beta)
        expl = get_expl(softplus_model, x, args.method, desired_index=idx, smooth=args.smooth, abs_value=False).detach().cpu()
        pdt = model(x)[:, idx]
        norm = np.linalg.norm(expl)
        if norm <=1e-4:
            print("small norm!!!")
        infds.append(get_exp_infid(x, model, expl, idx, pdt.detach().cpu().numpy(),
                                   binary_I=args.binary_I, pert=args.infidelity_pert))
        senss.append(get_exp_sens(x, softplus_model, expl, args.method, idx, pdt, sg_r=1.0, sg_N=50, sen_r=0.1, sen_N=50,
                                  norm=norm, smooth=args.smooth, binary_I=args.binary_I))
    all_infds = all_infds + np.array(infds)
    all_senss = all_senss + np.array(senss)
    i+=1
    if i%10==0:
        print("done!")

np.save(result_path+"infds.npy", all_infds)
np.save(result_path+"senss.npy", all_senss)
print("run time: ", time.time()-t0)
