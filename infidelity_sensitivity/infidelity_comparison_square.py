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
sys.path.append("../../pytorch-cifar/models/")
from vgg import VGG
import argparse
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--output_dir', type=str, default='results_square/', help='directory to save results to')
argparser.add_argument('--infidelity_pert', type=str, default="Square", help='infidelity perturbation')
argparser.add_argument('--binary_I',type=bool,
                        help="whether to multiply the explanation with input or not", default=True)
argparser.add_argument('--softplus',type=bool,
                        help="whether to use beta-smoothing or not", default=False)
argparser.add_argument('--smooth',type=int,
                        help="whether to use smooth gradient or not", default=0)
args = argparser.parse_args()
print(args.smooth)
print(args.softplus)
if args.method == "saliency" and args.smooth == 1:
    if args.infidelity_pert=="Gaussian":
        result_path = args.output_dir+"smooth_grad/"
    elif args.infidelity_pert=="Square":
        result_path = args.output_dir+"smooth_grad_square_pert/"
    else:
        raise ValueError("wrong perturbation type.")
elif args.softplus == True:
    if args.infidelity_pert=="Gaussian":
        result_path = args.output_dir+"softplus/"
    elif args.infidelity_pert=="Square":
        result_path = args.output_dir+"softplus_square_pert/"
    else:
        raise ValueError("wrong perturbation type.")
else:
    if args.infidelity_pert=="Gaussian":
        result_path = args.output_dir+args.method+"/"
    elif args.infidelity_pert=="Square":
        result_path = args.output_dir+args.method+"_square_pert/"
    else:
        raise ValueError("wrong perturbation type.")
print(result_path)
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
normalizer = mister_ed_utils.DifferentiableNormalize(mean=data_mean,
                                               std=data_std)
#################################
seed = 72+1
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
        batch_size=32,
        shuffle=True
    )
dataiter = iter(test_loader)
#################################
# seed = 93
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# #################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cifar_test = torchvision.datasets.CIFAR10(root="../notebooks/data", train=False, download=True,
#                                           transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(
#         cifar_test,
#         batch_size=8,
#         shuffle=False
#     )
# #################################
# normalizer = mister_ed_utils.DifferentiableNormalize(mean=[0.4914, 0.4822, 0.4465],
#                                                std=[0.2023, 0.1994, 0.2010])
#################################
model = torchvision.models.vgg16(pretrained=True).to(device)
# model = VGG("VGG16")
# model.load_state_dict(torch.load("../notebooks/models/VGG16_cifar.pth")["net"])
# model = model.to(device)
model = model.eval()
if args.softplus:
    softplus_model = convert_relu_to_softplus(model, beta=0.8)
    softplus_model = softplus_model.eval()
    model = torchvision.models.vgg16(pretrained=True).to(device)
#     model = VGG("VGG16")
#     model.load_state_dict(torch.load("../notebooks/models/VGG16_cifar.pth")["net"])
#     model = model.to(device)
    model = model.eval()
#################################
t0 = time.time()
infds = []
senss = []
i=0
for images, labels in test_loader:
    ind = np.random.randint(len(labels))
    x = normalizer.forward(images[ind:ind+1]).to(device)
    idx = labels[ind].to(device)
    sigma = (torch.max(x) - torch.min(x)).item() * 0.2
    if args.softplus:
        expl = get_expl(softplus_model, x, args.method, desired_index=idx, smooth=args.smooth, sigma=sigma, abs_value=False).detach().cpu()
    else:
        expl = get_expl(model, x, args.method, desired_index=idx, smooth=args.smooth, sigma=sigma, abs_value=False).detach().cpu()
    pdt = model(x)[:, idx]
    norm = np.linalg.norm(expl)
    infds.append(get_exp_infid(x, model, expl, idx, pdt.detach().cpu().numpy(),
                               binary_I=args.binary_I, pert=args.infidelity_pert))
    if args.softplus:
        senss.append(get_exp_sens(x, softplus_model, expl, args.method, idx, pdt, sg_r=sigma, sg_N=50, sen_r=0.1, sen_N=50,
                                  norm=norm, smooth=args.smooth, binary_I=args.binary_I))
    else:
        senss.append(get_exp_sens(x, model, expl, args.method, idx, pdt, sg_r=sigma, sg_N=50, sen_r=0.1, sen_N=50,
                                  norm=norm, smooth=args.smooth, binary_I=args.binary_I))
    i+=1
    if i%10==0:
        print("done!")
    if i==100:
        break

np.save(result_path+"infds_100.npy", np.array(infds))
np.save(result_path+"senss_100.npy", np.array(senss))

print("run time: ", time.time()-t0)
