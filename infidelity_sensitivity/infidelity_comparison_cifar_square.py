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
from resnet import ResNet18
import argparse
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--output_dir', type=str, default='results_cifar_square/', help='directory to save results to')
argparser.add_argument('--infidelity_pert', type=str, default="Square", help='infidelity perturbation')
argparser.add_argument('--binary_I',type=bool,
                        help="whether to multiply the explanation with input or not", default=True)
argparser.add_argument('--model_path', type=str, default='../notebooks/models/RN18_standard.pth',
                       help='path to the pretrained model weigths')
args = argparser.parse_args()

result_path = args.output_dir
print(result_path)
#################################
seed = 93
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cifar_test = torchvision.datasets.CIFAR10(root="../notebooks/data", train=False, download=True,
                                          transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
        cifar_test,
        batch_size=8,
        shuffle=False
    )
#################################
dataiter = iter(test_loader)
#################################
# ResNet 18 for both CURE and adv training
model = ResNet18()
model.load_state_dict(torch.load(args.model_path)["net"])
model = model.eval().to(device)
####
normalizer = mister_ed_utils.DifferentiableNormalize(mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010])
#################################
t0 = time.time()
infds = []
senss = []
i=0
for images, labels in test_loader:
    ind = np.random.randint(len(labels))
    x = normalizer.forward(images[ind:ind+1]).to(device)
    idx = labels[ind].to(device)
    expl = get_expl(model, x, args.method, desired_index=idx, abs_value=False).detach().cpu()
    pdt = model(x)[:, idx]
    norm = np.linalg.norm(expl)
    infds.append(get_exp_infid(x, model, expl, idx, pdt.detach().cpu().numpy(),
                               binary_I=args.binary_I, pert=args.infidelity_pert))
    senss.append(get_exp_sens(x, model, expl, args.method, idx, pdt, sg_r=0.0, sg_N=50, sen_r=0.1, sen_N=50,
                                  norm=norm, smooth=False, binary_I=args.binary_I))
    i+=1
    if i%50==0:
        print("done!")
    if i==200:
        break

np.save(result_path+"infds.npy", np.array(infds))
np.save(result_path+"senss.npy", np.array(senss))

print("run time: ", time.time()-t0)
