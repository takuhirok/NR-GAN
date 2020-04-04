import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def additive_gaussian_reparameterize(input, images, eps=None):
    if eps is None:
        eps = torch.randn_like(images)
    return eps * input


def multiplicative_gaussian_reparameterize(input,
                                           images,
                                           eps=None,
                                           image_range=(-1, 1)):
    mean = np.mean(image_range)
    scale = image_range[1] - image_range[0]
    if eps is None:
        eps = torch.randn_like(images)
    return (eps * input * ((images.detach() - mean) / scale + 0.5)) * scale


def poisson_reparameterize(input, images, eps=None, image_range=(-1, 1)):
    mean = np.mean(image_range)
    scale = image_range[1] - image_range[0]
    if eps is None:
        eps = torch.randn_like(images)
    return (eps * input *
            torch.sqrt((images.detach() - mean) / scale + 0.5)) * scale


def rotate(input):
    batch_size = input.size(0)
    label = torch.randint(4, size=(batch_size, )).to(input.device)
    output = input.clone()
    output[label == 1] = input[label == 1].transpose(2, 3).flip(3)
    output[label == 2] = input[label == 2].flip(2).flip(3)
    output[label == 3] = input[label == 3].transpose(2, 3).flip(2)
    return output


def shuffle_channel(input):
    batch_size, channel, height, width = input.size()
    perm = torch.cat(
        [torch.randperm(channel) + channel * i for i in range(batch_size)])
    output = input.clone()
    output = output.view(batch_size * channel, height, width)
    output = output[perm]
    output = output.view(batch_size, channel, height, width)
    return output


def inverse_color(input):
    batch_size, channel, height, width = input.size()
    output = input.clone()
    output = output.view(batch_size * channel, height, width)
    label = torch.randint(2, size=(batch_size * channel, )).to(input.device)
    output[label == 1] = -input.view(batch_size * channel, height,
                                     width)[label == 1]
    output = output.view(batch_size, channel, height, width)
    return output


def blurvh(input, kernel_size=3):
    filt_v = Blur2d(input.size(1), kernel_size, 'v').to(input.device)
    filt_h = Blur2d(input.size(1), kernel_size, 'h').to(input.device)
    return torch.cat((filt_v(input), filt_h(input)), dim=1)


class Blur2d(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=3,
                 direction='vh',
                 padding_mode='zeros'):
        super(Blur2d, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.direction = direction
        self.padding_mode = padding_mode

        if kernel_size == 1:
            weight = torch.Tensor([1.])
        elif kernel_size == 2:
            weight = torch.Tensor([1., 1.])
        elif kernel_size == 3:
            weight = torch.Tensor([1., 2., 1.])
        elif kernel_size == 4:
            weight = torch.Tensor([1., 3., 3., 1.])
        elif kernel_size == 5:
            weight = torch.Tensor([1., 4., 6., 4., 1.])
        elif kernel_size == 6:
            weight = torch.Tensor([1., 5., 10., 10., 5., 1.])
        elif kernel_size == 7:
            weight = torch.Tensor([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise NotImplementedError

        if direction == 'vh':
            weight = weight[:, None] * weight[None, :]
        elif direction == 'v':
            weight = weight[:, None]
        elif direction == 'h':
            weight = weight[None, :]
        else:
            raise ValueError('Unknown direction: {}'.format(direction))
        weight = weight / weight.sum()
        height, width = weight.size()

        self.register_buffer('weight',
                             weight[None, None, :, :].repeat(channel, 1, 1, 1))
        padding = (
            int((width - 1) / 2.),
            int(np.ceil((width - 1) / 2.)),
            int((height - 1) / 2.),
            int(np.ceil((height - 1) / 2.)),
        )
        if self.padding_mode == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif self.padding_mode == 'reflection':
            self.pad = nn.ReflectionPad2d(padding)
        elif self.padding_mode == 'replication':
            self.pad = nn.ReplicationPad2d(padding)

    def forward(self, input):
        return F.conv2d(self.pad(input), self.weight, groups=input.shape[1])


def gradient_penalty_regularization(x, netD):
    device = x.device
    with torch.enable_grad():
        x = x.detach()
        x.requires_grad_(True)
        output = netD(x)
        grad_output = torch.ones(output.size()).to(device)
        grad = torch.autograd.grad(outputs=output,
                                   inputs=x,
                                   grad_outputs=grad_output,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        grad = grad.contiguous().view(grad.size(0), -1)
        reg = grad.pow(exponent=2).sum(dim=1).mean()
    return reg


def diversity_sensitive_regularization(x, z, eps=1e-12):
    B = x.size(0) // 2
    if isinstance(x, (list, tuple)):
        x_dist = 0
        for i in range(len(x)):
            x_dist = x_dist + l1_dist(x[i][:B], x[i][B:]) / len(x)
    else:
        x_dist = l1_dist(x[:B], x[B:])
    z_dist = l1_dist(z[:B], z[B:])
    loss = -torch.mean(x_dist / (z_dist + eps))
    return loss


def l1_dist(input, target):
    return torch.mean(torch.abs(input - target).view(input.size(0), -1), dim=1)


def update_average(model_target, model_source, beta):
    with torch.no_grad():
        param_dict_source = dict(model_source.named_parameters())
        for p_name, p_target in model_target.named_parameters():
            p_source = param_dict_source[p_name]
            assert (p_source is not p_target)
            p_target.copy_(beta * p_target + (1. - beta) * p_source)

        buf_dict_source = dict(model_source.named_buffers())
        for b_name, b_target in model_target.named_buffers():
            b_source = buf_dict_source[b_name]
            assert (b_source is not b_target)
            if 'running_mean' in b_name or 'running_var' in b_name:
                b_target.copy_(b_source)
