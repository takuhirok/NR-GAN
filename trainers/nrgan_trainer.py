import os

import torch
import torch.nn.functional as F
from torchvision import utils

from . import util


class NRGANTrainer():
    def __init__(self, *args, **kwargs):
        self.iterator = kwargs.pop('iterator')
        (self.netG, self.netD, self.netGn) = kwargs.pop('models')
        (self.netG_test, self.netGn_test) = kwargs.pop('models_test',
                                                       (None, None))
        self.noise_measure, = kwargs.pop('measures', (None, ))
        (self.optimizerG, self.optimizerD,
         self.optimizerGn) = kwargs.pop('optimizers')
        self.lr_schedulers = kwargs.pop('lr_schedulers')
        self.batch_size = kwargs.pop('batch_size')
        self.g_bs_multiple = kwargs.pop('g_bs_multiple', 1)
        self.num_critic = kwargs.pop('num_critic', 1)
        (self.lambda_r1, self.lambda_ds) = kwargs.pop('lambdas', (0., 0.))
        (self.g_model_average,
         self.gn_model_average) = kwargs.pop('model_averages', (False, False))
        self.model_average_beta = kwargs.pop('model_average_beta', 0.999)
        self.image_range = kwargs.pop('image_range', (-1, 1))
        self.implicit = kwargs.pop('implicit', False)
        self.prior = kwargs.pop('prior', None)
        self.rotation = kwargs.pop('rotation', False)
        self.channel_shuffle = kwargs.pop('channel_shuffle', False)
        self.color_inversion = kwargs.pop('color_inversion', False)
        self.blurvh = kwargs.pop('blurvh', False)
        self.clip_range = kwargs.pop('clip_range', (-1, 1))
        self.device = kwargs.pop('device')

        self.loss = {}
        self.iteration = 0

    def update(self):
        netG, netD, netGn = self.netG, self.netD, self.netGn
        netG.train()
        netD.train()
        if netGn is not None:
            netGn.train()

        (optimizerG, optimizerD,
         optimizerGn) = (self.optimizerG, self.optimizerD, self.optimizerGn)

        if self.iteration > 0:
            # Train G
            g_batch_size = self.g_bs_multiple * self.batch_size

            zx = netG.sample_z(g_batch_size).to(self.device)
            x_fake = netG(zx)

            if netGn is not None:
                zn = netGn.sample_z(g_batch_size).to(self.device)
                if self.implicit:
                    zn[:, :netG.latent_dim] = zx
                n_fake = netGn(zn)
                if self.prior == 'additive_gaussian':
                    n_fake = util.additive_gaussian_reparameterize(
                        n_fake, x_fake)
                elif self.prior == 'multiplicative_gaussian':
                    n_fake = util.multiplicative_gaussian_reparameterize(
                        n_fake, x_fake, image_range=self.image_range)
                elif self.prior == 'poisson':
                    n_fake = util.poisson_reparameterize(
                        n_fake, x_fake, image_range=self.image_range)
                elif self.prior is not None:
                    raise ValueError('Unknown prior: {}'.format(self.prior))
                if self.lambda_ds > 0:
                    gn_loss_ds = (
                        self.lambda_ds *
                        util.diversity_sensitive_regularization(n_fake, zn))
                if self.rotation:
                    n_fake = util.rotate(n_fake)
                if self.channel_shuffle:
                    n_fake = util.shuffle_channel(n_fake)
                if self.color_inversion:
                    n_fake = util.inverse_color(n_fake)
                x_fake = x_fake + n_fake
            if self.noise_measure is not None:
                x_fake = self.noise_measure(x_fake)

            if self.clip_range is not None:
                x_fake = torch.clamp(x_fake, *self.clip_range)
            if self.blurvh:
                x_fake = util.blurvh(x_fake)
            out_dis = netD(x_fake)
            g_loss_fake = F.binary_cross_entropy_with_logits(
                out_dis,
                torch.ones(g_batch_size).to(self.device))

            g_loss = 0
            g_loss = g_loss + g_loss_fake
            if netGn is not None and self.lambda_ds > 0:
                g_loss = g_loss + gn_loss_ds
            netG.zero_grad()
            if netGn is not None:
                netGn.zero_grad()
            g_loss.backward()
            optimizerG.step()
            if netGn is not None:
                optimizerGn.step()

            self.loss['G/loss_fake'] = g_loss_fake.item()
            if netGn is not None and self.lambda_ds > 0:
                self.loss['Gn/loss_ds'] = gn_loss_ds.item()

            if self.g_model_average:
                util.update_average(self.netG_test, netG,
                                    self.model_average_beta)
            if netGn is not None and self.gn_model_average:
                util.update_average(self.netGn_test, netGn,
                                    self.model_average_beta)

            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()

        # Train D
        for i in range(self.num_critic):
            image_real, _ = next(self.iterator)
            x_real = image_real.to(self.device)
            if self.clip_range is not None:
                x_real = torch.clamp(x_real, *self.clip_range)
            if self.blurvh:
                x_real = util.blurvh(x_real)
            out_dis = netD(x_real)
            d_loss_real = F.binary_cross_entropy_with_logits(
                out_dis,
                torch.ones(self.batch_size).to(self.device))

            if self.lambda_r1 > 0:
                d_loss_r1 = (
                    self.lambda_r1 *
                    util.gradient_penalty_regularization(x_real, netD))

            zx = netG.sample_z(self.batch_size).to(self.device)
            x_fake = netG(zx)

            if netGn is not None:
                zn = netGn.sample_z(self.batch_size).to(self.device)
                if self.implicit:
                    zn[:, :netG.latent_dim] = zx
                n_fake = netGn(zn)
                if self.prior == 'additive_gaussian':
                    n_fake = util.additive_gaussian_reparameterize(
                        n_fake, x_fake)
                elif self.prior == 'multiplicative_gaussian':
                    n_fake = util.multiplicative_gaussian_reparameterize(
                        n_fake, x_fake, image_range=self.image_range)
                elif self.prior == 'poisson':
                    n_fake = util.poisson_reparameterize(
                        n_fake, x_fake, image_range=self.image_range)
                if self.rotation:
                    n_fake = util.rotate(n_fake)
                if self.channel_shuffle:
                    n_fake = util.shuffle_channel(n_fake)
                if self.color_inversion:
                    n_fake = util.inverse_color(n_fake)
                x_fake = x_fake + n_fake
            if self.noise_measure is not None:
                x_fake = self.noise_measure(x_fake)

            if self.clip_range is not None:
                x_fake = torch.clamp(x_fake, *self.clip_range)
            if self.blurvh:
                x_fake = util.blurvh(x_fake)
            out_dis = netD(x_fake.detach())
            d_loss_fake = F.binary_cross_entropy_with_logits(
                out_dis,
                torch.zeros(self.batch_size).to(self.device))

            d_loss = 0
            d_loss = d_loss + d_loss_real + d_loss_fake
            if self.lambda_r1 > 0:
                d_loss = d_loss + d_loss_r1
            netD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            self.loss['D/loss_real'] = d_loss_real.item()
            self.loss['D/loss_fake'] = d_loss_fake.item()
            if self.lambda_r1 > 0:
                self.loss['D/loss_r1'] = d_loss_r1.item()
            self.loss['D/loss'] = d_loss.item()

        self.iteration += 1

    def get_current_loss(self):
        return self.loss


class NRGANVisualizer():
    def __init__(self,
                 netG,
                 netGn,
                 device,
                 out,
                 implicit=True,
                 prior=None,
                 rotation=False,
                 channel_shuffle=False,
                 color_inversion=False,
                 num_columns=10,
                 num_rows=None,
                 image_range=(-1, 1),
                 prefix='samples'):
        self.netG = netG
        self.netGn = netGn
        self.device = device
        self.out = out
        self.implicit = implicit
        self.prior = prior
        self.rotation = rotation
        self.channel_shuffle = channel_shuffle
        self.color_inversion = color_inversion
        self.num_columns = num_columns
        if num_rows is None:
            self.num_rows = num_columns
        else:
            self.num_rows = num_rows
        self.image_range = image_range
        self.prefix = prefix

        self.fixed_zx = netG.sample_z(self.num_columns *
                                      self.num_rows).to(device)
        if netGn is not None:
            self.fixed_zn = netGn.sample_z(self.num_columns *
                                           self.num_rows).to(device)
            if implicit:
                self.fixed_zn[:, :netG.latent_dim] = self.fixed_zx
            if prior is not None:
                with torch.no_grad():
                    x = netG(self.fixed_zx)
                    self.fixed_eps = torch.randn_like(x)

    def save_image(self, x, prefix, iteration, range):
        utils.save_image(x.detach(),
                         os.path.join(self.out,
                                      '%s_iter_%d.png' % (prefix, iteration)),
                         self.num_columns,
                         0,
                         normalize=True,
                         range=range)

    def visualize(self, iteration):
        (netG, netGn) = (self.netG, self.netGn)
        netG.eval()
        if netGn is not None:
            netGn.eval()

        with torch.no_grad():
            x = netG(self.fixed_zx)
            self.save_image(x, self.prefix, iteration, self.image_range)

            if netGn is not None:
                n = netGn(self.fixed_zn)
                if self.prior == 'additive_gaussian':
                    n = util.additive_gaussian_reparameterize(
                        n, x, self.fixed_eps)
                elif self.prior == 'multiplicative_gaussian':
                    n = util.multiplicative_gaussian_reparameterize(
                        n, x, self.fixed_eps, self.image_range)
                elif self.prior == 'poisson':
                    n = util.poisson_reparameterize(n, x, self.fixed_eps,
                                                    self.image_range)
                if self.rotation:
                    n = util.rotate(n)
                if self.channel_shuffle:
                    n = util.shuffle_channel(n)
                if self.color_inversion:
                    n = util.inverse_color(n)
                self.save_image(n, self.prefix + '_noise', iteration,
                                self.image_range)
                x = x + n
                self.save_image(x, self.prefix + '_noisy', iteration,
                                self.image_range)
