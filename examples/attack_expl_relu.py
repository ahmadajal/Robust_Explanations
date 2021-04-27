# EXTERNAL LIBRARIES
import numpy as np
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("../attacks/")
# mister_ed
import mister_ed.loss_functions as lf
import mister_ed.utils.pytorch_utils as utils
import mister_ed.utils.image_utils as img_utils
import mister_ed.cifar10.cifar_loader as cifar_loader
import mister_ed.cifar10.cifar_resnets as cifar_resnets
import mister_ed.adversarial_training as advtrain
import mister_ed.utils.checkpoints as checkpoints
import mister_ed.adversarial_perturbations as ap
import mister_ed.adversarial_attacks as aa
import mister_ed.spatial_transformers as st
import mister_ed.config as config
# ReColorAdv
import recoloradv.perturbations as pt
import recoloradv.color_transformers as ct
import recoloradv.color_spaces as cs
from recoloradv import norms
from utils import load_image, torch_to_image, get_expl, convert_relu_to_softplus, plot_overview, UniGrad
# explanation
import sys
# sys.path.append("../Spatial_transform/ST_ADV_exp_imagenet/")
sys.path.append("../PerceptualSimilarity/") # for LPIPS similarity
sys.path.append("../Perc-Adversarial/") # for perceptual color distance regulariation - https://github.com/ZhengyuZhao/PerC-Adversarial
import lpips
from differential_color_functions import rgb2lab_diff, ciede2000_diff
# from nn.utils import load_image, torch_to_image, clamp
# from nn.enums import ExplainingMethod
# from nn.networks import ExplainableNet
# from nn.utils import get_expl, heatmap_to_image, plot_overview
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=1000, help='number of iterations')
argparser.add_argument('--img', type=str, default='data/collie4.jpeg', help='image net file to run attack on')
argparser.add_argument('--target_img', type=str, default='data/tiger_cat.jpeg',
                       help='imagenet file used to generate target expl')
argparser.add_argument('--lr', type=float, default=0.0002, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_expl_relu/', help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='lrp')
argparser.add_argument('--smooth',type=bool, help="wether to use the smooth explanation or not", default=False)
argparser.add_argument('--additive_lp_bound', type=float, default=0.03, help='l_p bound for additive attack')
argparser.add_argument('--stadv_lp_bound', type=float, default=0.05, help='l_p bound for spatial transformation')
argparser.add_argument('--norm_weights', nargs=3, default=[1.0, 1.0, 0.0], type=float,
                        help='norm weights for combining smooth loss for each attack')
argparser.add_argument('--lpips_reg', type=float, default=0.0, help='LPIPS regularizer coeff. if 0 then there will be no LPIPS regularization')
argparser.add_argument('--ciede2000_reg', type=float, default=0.0, help='ciede2000 distance regularizer coeff. if 0 then there will be no such regularization')
argparser.add_argument('--attack_type', nargs=3, default=[1, 1, 1], type=int,
                       help='type of the attack. only 0 and 1 values are accepted.\
                       the order is [ReColorAdv, STadv, additive]')
argparser.add_argument('--early_stop_for', type=str, default=None, help='eraly stop for part of the loss')
argparser.add_argument('--early_stop_value', type=float, default=10.0, help='stop the optimization if \
                        (part of) the loss dropped below a certain level.')
args = argparser.parse_args()
print(args.smooth)
a = np.array(args.attack_type)
if ((a != 1) & (a != 0)).any():
    raise ValueError("only 0 or 1 values are accepted for attack type combination")
from PIL import Image
im = Image.open("../sample_imagenet/sample_0.jpg")
examples = torchvision.transforms.ToTensor()(
        torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im)))
examples = examples.unsqueeze(0)
labels = torch.tensor([17])

model = torchvision.models.vgg16(pretrained=True)
# we need to substitute the ReLus with softplus to avoid zero second derivative
model = convert_relu_to_softplus(model, beta=100)
####
model = model.eval()
####
normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

if utils.use_gpu():
    examples = examples.cuda()
    labels = labels.cuda()
    model.cuda()

# sigma for the smooth grad and uniform grad methods
sigma = (torch.max(normalizer.forward(examples)) - torch.min(normalizer.forward(examples))).item() * 0.2
print("sigma: ", sigma)
## expl loss
class EXPL_Loss_mse(lf.PartialLoss):
    def __init__(self, classifier, target_expl, method, normalizer=None, smooth=False):
        super(EXPL_Loss_mse, self).__init__()
        self.classifier = classifier
        self.target_expl = target_expl
        self.method = method
        self.normalizer = normalizer
        self.smooth = smooth
        self.nets.append(self.classifier)


    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get adversarial expl
        adv_expl = get_expl(self.classifier, classifier_in, self.method,
                            desired_index=labels, smooth=self.smooth, sigma=sigma, normalize=True)
        ####temp#####
        # adv_expl, all_adv_expl = UniGrad(self.classifier, classifier_in, target=labels,
        #                                 num_steps=5, sigma=sigma, return_noisy_expls=True)
        # loss_expl = 0
        # for expl in all_adv_expl:
        #     expl = torch.sum(expl.squeeze(), 0, True)
        #     expl = expl / torch.sum(expl)
        #     loss_expl = loss_expl + F.mse_loss(expl, self.target_expl)
        # loss_expl = loss_expl / len(all_adv_expl)
        #############
        loss_expl = F.mse_loss(adv_expl, self.target_expl)
        print("expl loss:", loss_expl.item())
        return loss_expl

## output loss
class OUTPUT_Loss_mse(lf.PartialLoss):
    def __init__(self, classifier, org_logits, normalizer=None):
        super(OUTPUT_Loss_mse, self).__init__()
        self.classifier = classifier
        self.org_logits = org_logits
        self.normalizer = normalizer
        self.nets.append(self.classifier)


    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get target expl
        loss_output = F.mse_loss(F.softmax(self.classifier(classifier_in), dim=1), self.org_logits)
        print("output loss:", loss_output.item())
        return loss_output

# target expl
im = Image.open("../sample_imagenet/sample_0_target.jpg")
target_examples = torchvision.transforms.ToTensor()(
        torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im)))
target_examples = target_examples.unsqueeze(0)
if utils.use_gpu():
    target_examples = target_examples.cuda()
target_label = model(normalizer.forward(target_examples)).argmax()
# we always use the saliency as the target explanation map
target_expl = get_expl(model, normalizer.forward(target_examples), "saliency",
                        desired_index=target_label, smooth=False, sigma=sigma, normalize=True)
target_expl = target_expl.detach()
# original explanation and logits - need not to be normalized as it is not part of the objective func
org_expl = get_expl(model, normalizer.forward(examples),
                         args.method, desired_index=target_label, smooth=args.smooth, sigma=sigma)
org_expl = org_expl.detach().cpu()
org_logits = F.softmax(model(normalizer.forward(examples)), dim=1)
org_logits = org_logits.detach()

# This threat model defines the regularization parameters of the attack.
recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
    'xform_class': ct.FullSpatial,
    'cspace': cs.CIELUVColorSpace(), # controls the color space used
    'lp_style': 'inf',
    'lp_bound': [0.06, 0.06, 0.06],  # [epsilon_1, epsilon_2, epsilon_3]
    'xform_params': {
      'resolution_x': 16,            # R_1
      'resolution_y': 32,            # R_2
      'resolution_z': 32,            # R_3
    },
    'use_smooth_loss': True,
})

# we define the additive threat model.
additive_threat = ap.ThreatModel(ap.DeltaAddition, {
   'lp_style': 'inf',
   'lp_bound': args.additive_lp_bound,
})

# spatial transformation attack
stadv_threat = ap.ThreatModel(
                ap.ParameterizedXformAdv,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=args.stadv_lp_bound,
                    xform_class=st.FullSpatial,
                    use_stadv=True,
                ))

all_threats = [recoloradv_threat, stadv_threat, additive_threat]
threat_comb = []
norm_weights = []
for ind, threat in enumerate(all_threats):
    if args.attack_type[ind] == 1:
        threat_comb.append(all_threats[ind])
        norm_weights.append(args.norm_weights[ind])
print("attack: ", threat_comb)
# Combine all the threat models.
combined_threat = ap.ThreatModel(
    ap.SequentialPerturbation,
    threat_comb,
    ap.PerturbationParameters(norm_weights=norm_weights),
)

# Again, define the optimization terms.
# explanation loss
expl_loss = EXPL_Loss_mse(model, target_expl, args.method, normalizer, smooth=args.smooth)

# output loss
output_loss = OUTPUT_Loss_mse(model, org_logits, normalizer)

# temp: LPIPS loss to ensure visual similarity
lpips_loss = lf.LpipsRegularization(examples)

# temp: ciede2000 loss regularization to penalize perceptual color distance
ciede2000_loss = lf.ciede2000Regularization(examples)

# adv_loss = lf.CWLossF6(model, normalizer)
smooth_loss = lf.PerturbationNormLoss(lp=2)
attack_loss = lf.RegularizedLoss({'expl': expl_loss, 'out': output_loss, 'smooth': smooth_loss,
                                  'lpips': lpips_loss, 'ciede2000': ciede2000_loss},
                                 {'expl': 1e11 ,     'out': 1e5, 'smooth': 0.05,  # lambda = 0.05
                                  'lpips': args.lpips_reg, 'ciede2000': args.ciede2000_reg},
                                 negate=True) # Need this true for PGD type attacks

# Setup and run PGD over both perturbations at once.
pgd_attack_obj = aa.PGD(model, normalizer, combined_threat, attack_loss)
perturbation = pgd_attack_obj.attack(examples, labels, num_iterations=args.num_iter, signed=False,
                                     optimizer=optim.Adam, optimizer_kwargs={'lr': args.lr},
                                     verbose=True, early_stop_for=args.early_stop_for,
                                     early_stop_value=args.early_stop_value)
adv_expl = get_expl(model, normalizer.forward(perturbation.adversarial_tensors()), args.method,
                    desired_index=labels, smooth=args.smooth, sigma=sigma)
# get the target expl again without normalize to pass to the plot function
target_expl = get_expl(model, normalizer.forward(target_examples), "saliency",
                        desired_index=target_label, smooth=False, sigma=sigma)
s = "_smooth" if args.smooth else ""
plot_overview([normalizer.forward(target_examples),
               normalizer.forward(examples),
               normalizer.forward(perturbation.adversarial_tensors())], \
               [target_expl, org_expl, adv_expl], \
               data_mean, data_std, filename=f"{args.output_dir}overview_{args.method}{s}.png")
torch.save(perturbation.adversarial_tensors(), f"{args.output_dir}x_{args.method}{s}.pth") # this is the unnormalized tensor

print("attack type:", args.attack_type)
if sum(args.attack_type) <= 3:
    print(perturbation.layer_00.perturbation_norm(perturbation.adversarial_tensors(), 2))
if sum(args.attack_type) >= 2:
    print(perturbation.layer_01.perturbation_norm(perturbation.adversarial_tensors(), 2))
if sum(args.attack_type) == 3:
    print(perturbation.layer_02.perturbation_norm(perturbation.adversarial_tensors(), 2))
loss_lpips = lpips.LPIPS(net='vgg')
print("LPIPS: ", loss_lpips.forward(examples.cpu(), perturbation.adversarial_tensors().cpu()).item())
