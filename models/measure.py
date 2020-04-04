import cv2
import numpy as np

import torch
import torch.nn.functional as F


def additive_gaussian_noise_measure(input,
                                    noise_scale,
                                    noise_scale_high=None,
                                    image_range=(-1, 1),
                                    with_noise=False):
    if noise_scale_high is None:
        _noise_scale = noise_scale
    else:
        _noise_scale = torch.empty(input.size(0)).uniform_(
            noise_scale,
            noise_scale_high)[:, None, None, None].to(input.device)
    eps = torch.randn_like(input)
    noise = eps * _noise_scale / 255. * (image_range[1] - image_range[0])
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def local_gaussian_noise_measure(input,
                                 noise_scale,
                                 patch_size,
                                 noise_scale_high=None,
                                 patch_max_size=None,
                                 image_range=(-1, 1),
                                 with_noise=False):
    batch_size, _, height, width = input.shape
    patch = torch.zeros((batch_size, 1, height, width))
    for i in range(batch_size):
        if patch_max_size is None:
            patch_width = patch_size
            patch_height = patch_size
        else:
            patch_width = torch.randint(patch_size, patch_max_size + 1,
                                        (1, )).item()
            patch_height = torch.randint(patch_size, patch_max_size + 1,
                                         (1, )).item()
        x = torch.randint(0, width - patch_width + 1, (1, )).item()
        y = torch.randint(0, height - patch_height + 1, (1, )).item()
        patch[i][:, y:y + patch_height, x:x + patch_width] = 1
    patch = patch.to(input.device)
    noise = additive_gaussian_noise_measure(input,
                                            noise_scale,
                                            noise_scale_high,
                                            image_range=image_range,
                                            with_noise=True)[1]
    noise = noise * patch
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def uniform_noise_measure(input,
                          noise_scale,
                          noise_scale_high=None,
                          image_range=(-1, 1),
                          with_noise=False):
    if noise_scale_high is None:
        _noise_scale = noise_scale
    else:
        _noise_scale = torch.empty(input.size(0)).uniform_(
            noise_scale,
            noise_scale_high)[:, None, None, None].to(input.device)
    eps = (torch.rand_like(input) * 2.) - 1.
    noise = eps * _noise_scale / 255. * (image_range[1] - image_range[0])
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def mixture_noise_measure(input,
                          noise_scale_list,
                          mixture_rate_list,
                          image_range=(-1, 1),
                          with_noise=False):
    batch_size, channel, height, width = input.shape
    noise = [None] * batch_size
    for i in range(batch_size):
        noise[i] = torch.zeros((channel, height * width))
        perm = torch.randperm(height * width)
        rand = torch.rand(height * width)
        cumsum = np.cumsum([0] + mixture_rate_list)
        for j, noise_scale in enumerate(noise_scale_list):
            inds = (rand >= cumsum[j]) * (rand < cumsum[j + 1])
            if j == len(noise_scale_list) - 1:
                noise[i][:, perm[inds]] = (
                    (torch.rand(channel, torch.sum(inds)) * 2) -
                    1) * noise_scale / 255. * (image_range[1] - image_range[0])
            else:
                noise[i][:, perm[inds]] = torch.randn(
                    channel, torch.sum(inds)) * noise_scale / 255. * (
                        image_range[1] - image_range[0])
        noise[i] = noise[i].view(channel, height, width).to(input.device)
    noise = torch.stack(noise)
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def brown_gaussian_noise_measure(input,
                                 noise_scale,
                                 noise_scale_high=None,
                                 kernel_size=5,
                                 image_range=(-1, 1),
                                 with_noise=False):
    noise = additive_gaussian_noise_measure(input,
                                            noise_scale,
                                            noise_scale_high,
                                            image_range=image_range,
                                            with_noise=True)[1]
    padding = int((kernel_size - 1) / 2)
    kernel = torch.Tensor(
        cv2.getGaussianKernel(kernel_size, 0) *
        cv2.getGaussianKernel(kernel_size, 0).transpose()).to(input.device)
    kernel = kernel / torch.sqrt(torch.sum(kernel**2))
    kernel = kernel[None, None]
    kernel = kernel.expand(input.size(1), -1, -1, -1)
    noise = F.conv2d(noise,
                     kernel,
                     stride=1,
                     padding=padding,
                     groups=input.size(1))
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def additive_brown_gaussian_noise_measure(input,
                                          noise_scale,
                                          noise_scale_high=None,
                                          kernel_size=5,
                                          image_range=(-1, 1),
                                          with_noise=False):
    noise = additive_gaussian_noise_measure(input,
                                            noise_scale,
                                            noise_scale_high,
                                            image_range=image_range,
                                            with_noise=True)[1]
    padding = int((kernel_size - 1) / 2)
    kernel = torch.Tensor(
        cv2.getGaussianKernel(kernel_size, 0) *
        cv2.getGaussianKernel(kernel_size, 0).transpose()).to(input.device)
    kernel = kernel / torch.sqrt(torch.sum(kernel**2))
    kernel = kernel[None, None]
    kernel = kernel.expand(input.size(1), -1, -1, -1)
    noise = noise + F.conv2d(
        noise, kernel, stride=1, padding=padding, groups=input.size(1))
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def multiplicative_gaussian_noise_measure(input,
                                          multi_noise_scale,
                                          multi_noise_scale_high=None,
                                          image_range=(-1, 1),
                                          with_noise=False):
    mean = np.mean(image_range)
    scale = image_range[1] - image_range[0]
    if multi_noise_scale_high is None:
        _multi_noise_scale = multi_noise_scale
    else:
        _multi_noise_scale = torch.empty(input.size(0)).uniform_(
            multi_noise_scale,
            multi_noise_scale_high)[:, None, None, None].to(input.device)
    eps = torch.randn_like(input)
    noise = eps * _multi_noise_scale / 255. * (
        (input.detach() - mean) / scale + 0.5) * scale
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def additive_multiplicative_gaussian_noise_measure(input,
                                                   noise_scale,
                                                   multi_noise_scale,
                                                   noise_scale_high=None,
                                                   multi_noise_scale_high=None,
                                                   image_range=(-1, 1),
                                                   with_noise=False):
    noise_mg = multiplicative_gaussian_noise_measure(input,
                                                     multi_noise_scale,
                                                     multi_noise_scale_high,
                                                     image_range=image_range,
                                                     with_noise=True)[1]
    noise_ag = additive_gaussian_noise_measure(input,
                                               noise_scale,
                                               noise_scale_high,
                                               image_range=image_range,
                                               with_noise=True)[1]
    noise = noise_mg + noise_ag
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def poisson_noise_measure(input,
                          noise_lam,
                          noise_lam_high=None,
                          image_range=(-1, 1),
                          with_noise=False):
    mean = np.mean(image_range)
    scale = image_range[1] - image_range[0]
    if noise_lam_high is None:
        _noise_scale = np.sqrt(1. / noise_lam)
    else:
        _noise_lam = torch.empty(input.size(0)).uniform_(
            noise_lam, noise_lam_high)[:, None, None, None].to(input.device)
        _noise_scale = torch.sqrt(1. / _noise_lam)
    eps = torch.randn_like(input)
    noise = (eps * _noise_scale *
             torch.sqrt((input.detach() - mean) / scale + 0.5)) * scale
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output


def poisson_gaussian_noise_measure(input,
                                   noise_lam,
                                   noise_scale,
                                   noise_lam_high=None,
                                   noise_scale_high=None,
                                   image_range=(-1, 1),
                                   with_noise=False):
    noise_p = poisson_noise_measure(input,
                                    noise_lam,
                                    noise_lam_high,
                                    image_range=image_range,
                                    with_noise=True)[1]
    noise_ag = additive_gaussian_noise_measure(input,
                                               noise_scale,
                                               noise_scale_high,
                                               image_range=image_range,
                                               with_noise=True)[1]

    noise = noise_p + noise_ag
    output = input + noise
    if with_noise:
        return output, noise
    else:
        return output
