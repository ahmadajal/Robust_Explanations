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
sys.path.append("../attacks/")
from utils import torch_to_image, load_image,  convert_relu_to_softplus, change_beta_softplus, heatmap_to_image, get_expl

FORWARD_BZ = 50


def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize:end]
            out = model(tempinput.cuda())
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize:end]
            temp = model(tempinput.cuda()).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out

def set_zero_infid(array, size, point, pert):
    if pert == "Gaussian":
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])
    elif pert == "SHAP":
        nz_ind = np.nonzero(array)[0]
        nz_ind = np.arange(array.shape[0])
        num_nz = len(nz_ind)
        bb = 0
        while bb == 0 or bb == num_nz:
            aa = np.random.rand(num_nz)
            bb = np.sum(aa < 0.5)
        sample_ind = np.where(aa < 0.5)
        array[nz_ind[sample_ind]] = 0
        ind = np.zeros(array.shape)
        ind[nz_ind[sample_ind]] = 1
        return np.concatenate([array, ind])

def get_exp(ind, exp):
    return (exp[ind.astype(int)])

def get_imageset(image_copy, im_size, rads=[20]):
    rangelist = np.arange(np.prod(im_size)).reshape(im_size)
    width = im_size[1]
    height = im_size[2]
    ind = np.zeros(image_copy.shape)
    count = 0
    for rad in rads:
        for i in range(width - rad + 1):
            for j in range(height - rad + 1):
                ind[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 1
                image_copy[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 0
                count += 1
    return image_copy, ind

def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def shap_kernel(Z, X):
    M = X.shape[0]
    z_ = np.count_nonzero(Z)
    return (M-1) * 1.0 / (z_ * (M - 1 - z_) * nCr(M - 1, z_))

def get_exp_infid(image, model, exp, label, pdt, binary_I, pert):
    if len(image.size())==4:
        img_c = image.size()[1]
        img_h = image.size()[2]
        img_w = image.size()[3]
    else:
        img_c = image.size()[0]
        img_h = image.size()[1]
        img_w = image.size()[2]
    point = img_c*img_h*img_w #3*224*224
    total = (np.prod(exp.shape))
    num = 100
    if pert == 'Square':
        im_size = image.shape
        width = im_size[2]
        height = im_size[3]
        rads = np.arange(28, 30) + 1 #np.arange(9, 30) ImageNet , (15 cifar)
        num = 0
        for rad in rads:
            num += (width - rad + 1) * (height - rad + 1)
    exp = np.squeeze(exp)
    exp_copy = np.reshape(np.copy(exp), -1)
    image_copy = np.tile(np.reshape(np.copy(image.cpu()), -1), [num, 1])

    if pert == 'Gaussian':
        image_copy_ind = np.apply_along_axis(set_zero_infid, 1, image_copy, total, point, pert)
    elif pert == 'Square':
        image_copy, ind = get_imageset(image_copy, im_size[1:], rads=rads)

    if pert == 'Gaussian' and not binary_I:
        image_copy = image_copy_ind[:, :total]
        ind = image_copy_ind[:, total:total+point]
        rand = image_copy_ind[:, total+point:total+2*point]
        exp_sum = np.sum(rand*np.apply_along_axis(get_exp, 1, ind, exp_copy), axis=1)
        ks = np.ones(num)
    elif pert == 'Square' and binary_I:
        exp_sum = np.array([])
        for i in range(ind.shape[0]//10000 + 1):
            exp_sum = np.concatenate((exp_sum, np.sum(ind[i*10000:(i+1)*10000] * np.expand_dims(exp_copy, 0), axis=1)))
            gc.collect()
#         ks = np.apply_along_axis(shap_kernel, 1, ind, X=image.reshape(-1))
        ks = np.ones(num)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")

#     image_copy = np.reshape(image_copy, (num, 3, 224, 224))
    # divide the following lines in multiple steps to avoid memory overflow
    pdt_rm = np.array([])
    for i in range(image_copy.shape[0]//10000 + 1):
        image_copy_chunk = image_copy[i*10000:(i+1)*10000]
        image_copy_chunk = np.reshape(image_copy_chunk, (-1, img_c, img_h, img_w))
        image_v = Variable(torch.from_numpy(image_copy_chunk.astype(np.float32)).cuda(), requires_grad=False)
        out = forward_batch(model, image_v, FORWARD_BZ)
#         pdt_rm = (out[:, label])
        pdt_rm = np.concatenate((pdt_rm, out[:, label]))
        del image_v
        del image_copy_chunk
        gc.collect()
        torch.cuda.empty_cache()
    pdt_diff = pdt - pdt_rm

    # performs optimal scaling for each explanation before calculating the infidelity score
    beta = np.mean(ks*pdt_diff*exp_sum) / np.mean(ks*exp_sum*exp_sum)
    exp_sum *= beta
    infid = np.mean(ks*np.square(pdt_diff-exp_sum)) / np.mean(ks)
    # free up the memory.
    del image_copy
    gc.collect()
    torch.cuda.empty_cache()
    return infid

def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.normal(0, epsilon, size=dim)

def get_exp_sens(X, model, expl, expl_method, y, pdt, sg_r, sg_N, sen_r, sen_N, norm, smooth, binary_I=False):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
        X_noisy = X + sample
        expl_eps = get_expl(model, X_noisy, expl_method, y, sigma=sg_r,
                                     smooth=smooth, abs_value=False).detach().cpu()
        max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)) / norm
    return max_diff
