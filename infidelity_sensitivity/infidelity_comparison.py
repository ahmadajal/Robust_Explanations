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
argparser.add_argument('--smooth',type=bool, help="wether to use the smooth explanation or not", default=True)
argparser.add_argument('--output_dir', type=str, default='results_temp/', help='directory to save results to')
argparser.add_argument('--infidelity_pert', type=str, default="Gaussian", help='infidelity perturbation')
argparser.add_argument('--binary_I',type=bool,
                        help="whether to multiply the explanation with input or not", default=False)
args = argparser.parse_args()
if args.method == "saliency" and args.smooth == True:
    if args.infidelity_pert=="Gaussian":
        result_path = args.output_dir+"smooth_grad/"
    elif args.infidelity_pert=="Square":
        result_path = args.output_dir+"smooth_grad_square_pert/"
    else:
        raise ValueError("wrong perturbation type.")
else:
    if args.infidelity_pert=="Gaussian":
        result_path = args.output_dir+args.method+"/"
    elif args.infidelity_pert=="Square":
        result_path = args.output_dir+args.method+"_square_pert/"
    else:
        raise ValueError("wrong perturbation type.")
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
normalizer = mister_ed_utils.DifferentiableNormalize(mean=data_mean,
                                               std=data_std)
#################################
seed = 93
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#################################
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
model = torchvision.models.vgg16(pretrained=True).to(device)
model = model.eval()
#################################
t0 = time.time()
all_infds = 0
all_senss = 0
sigmas = []
i=0
for images, labels in test_loader:
    ind = np.random.randint(len(labels))
    x = normalizer.forward(images[ind:ind+1]).to(device)
    idx = labels[ind].to(device)
    sigmas.append((torch.max(x) - torch.min(x)).item() * 0.2)
    infds = []
    senss = []
    for s in range(0, 21):
        expl = get_expl(model, x, args.method, desired_index=idx, smooth=args.smooth, sigma=s*0.1, abs_value=False).detach().cpu()
        pdt = model(x)[:, idx]
        norm = np.linalg.norm(expl)
        infds.append(get_exp_infid(x, model, expl, idx, pdt.detach().cpu().numpy(),
                                   binary_I=args.binary_I, pert=args.infidelity_pert))
        senss.append(get_exp_sens(x, model, expl, args.method, idx, pdt, sg_r=s*0.1, sg_N=50, sen_r=0.1, sen_N=50,
                                  norm=norm, smooth=args.smooth, binary_I=args.binary_I))
    all_infds = all_infds + np.array(infds)
    all_senss = all_senss + np.array(senss)
    i+=1
    if i%10==0:
        print("done!")

np.save(result_path+"infds.npy", all_infds)
np.save(result_path+"senss.npy", all_senss)
np.save(result_path+"sigmas.npy", np.array(sigmas))

print("run time: ", time.time()-t0)
