import argparse
import copy
import functools
import os
import time

import torch
import torch.optim as optim
from torch.backends import cudnn
import torchvision.datasets
import torchvision.transforms as transforms

import datasets
from models import common, measure, net
from trainers import NRGANTrainer as Trainer
from trainers import NRGANVisualizer as Visualizer
from utils import util
from utils.logger import Logger


def main():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--dataset', type=str, default='CIFAR10AG')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--noise_scale', type=float, default=25.)
    parser.add_argument('--noise_scale_high', type=float, default=None)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--patch_max_size', type=int, default=None)
    parser.add_argument('--noise_scale_list',
                        type=float,
                        nargs='*',
                        default=[15, 25, 50])
    parser.add_argument('--mixture_rate_list',
                        type=float,
                        nargs='*',
                        default=[0.7, 0.2, 0.1])
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--noise_lam', type=float, default=30.)
    parser.add_argument('--noise_lam_high', type=float, default=None)
    parser.add_argument('--multi_noise_scale', type=float, default=25.)
    parser.add_argument('--multi_noise_scale_high', type=float, default=None)
    parser.add_argument('--no_clip', action='store_false', dest='clip')
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    # Model options
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--gn_train', action='store_true')
    parser.add_argument('--g_latent_dim', type=int, default=128)
    parser.add_argument('--g_image_size', type=int, default=32)
    parser.add_argument('--g_image_channels', type=int, default=3)
    parser.add_argument('--g_channels', type=int, default=128)
    parser.add_argument('--g_residual_factor', type=float, default=0.1)
    parser.add_argument('--d_channels', type=int, default=128)
    parser.add_argument('--d_residual_factor', type=float, default=0.1)
    parser.add_argument('--d_pooling', type=str, default='mean')
    # Measure option
    parser.add_argument('--noise_measure', action='store_true')
    # Training options
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--g_bs_multiple', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--num_critic', type=int, default=1)
    parser.add_argument('--lambda_r1', type=float, default=10.)
    parser.add_argument('--lambda_ds', type=float, default=0.02)
    parser.add_argument('--g_no_model_average',
                        action='store_false',
                        dest='g_model_average')
    parser.add_argument('--model_average_beta', type=float, default=0.999)
    parser.add_argument('--implicit', action='store_true')
    parser.add_argument('--prior', type=str, default=None)
    parser.add_argument('--rotation', action='store_true')
    parser.add_argument('--channel_shuffle', action='store_true')
    parser.add_argument('--color_inversion', action='store_true')
    parser.add_argument('--blurvh', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=200000)
    parser.add_argument('--num_iterations_decay', type=int, default=0)
    # Output options
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--snapshot_interval', type=int, default=5000)
    parser.add_argument('--visualize_interval', type=int, default=5000)
    parser.add_argument('--num_columns', type=int, default=10)
    args = parser.parse_args()

    # Set up options
    args.image_range = (-1, 1)
    if args.clip:
        args.clip_range = args.image_range
    else:
        args.clip_range = None

    def normalize(x):
        x = 2 * ((x * 255. / 256.) - .5)
        x += torch.zeros_like(x).uniform_(0, 1. / 128)
        return x

    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    # Set up dataset
    if args.dataset == 'CIFAR10':
        Dataset = torchvision.datasets.CIFAR10
    elif args.dataset == 'CIFAR10AG':
        Dataset = functools.partial(datasets.CIFAR10AdditiveGaussianNoise,
                                    noise_scale=args.noise_scale,
                                    noise_scale_high=args.noise_scale_high,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10LG':
        Dataset = functools.partial(datasets.CIFAR10LocalGaussianNoise,
                                    noise_scale=args.noise_scale,
                                    patch_size=args.patch_size,
                                    noise_scale_high=args.noise_scale_high,
                                    patch_max_size=args.patch_max_size,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10U':
        Dataset = functools.partial(datasets.CIFAR10UniformNoise,
                                    noise_scale=args.noise_scale,
                                    noise_scale_high=args.noise_scale_high,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10MIX':
        Dataset = functools.partial(datasets.CIFAR10MixtureNoise,
                                    noise_scale_list=args.noise_scale_list,
                                    mixture_rate_list=args.mixture_rate_list,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10BG':
        Dataset = functools.partial(datasets.CIFAR10BrownGaussianNoise,
                                    noise_scale=args.noise_scale,
                                    noise_scale_high=args.noise_scale_high,
                                    kernel_size=args.kernel_size,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10ABG':
        Dataset = functools.partial(datasets.CIFAR10AdditiveBrownGaussianNoise,
                                    noise_scale=args.noise_scale,
                                    noise_scale_high=args.noise_scale_high,
                                    kernel_size=args.kernel_size,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10MG':
        Dataset = functools.partial(
            datasets.CIFAR10MultiplicativeGaussianNoise,
            multi_noise_scale=args.multi_noise_scale,
            multi_noise_scale_high=args.multi_noise_scale_high,
            clip=args.clip,
            seed=args.data_seed)
    elif args.dataset == 'CIFAR10AMG':
        Dataset = functools.partial(
            datasets.CIFAR10AdditiveMultiplicativeGaussianNoise,
            noise_scale=args.noise_scale,
            multi_noise_scale=args.multi_noise_scale,
            noise_scale_high=args.noise_scale_high,
            multi_noise_scale_high=args.multi_noise_scale_high,
            clip=args.clip,
            seed=args.data_seed)
    elif args.dataset == 'CIFAR10P':
        Dataset = functools.partial(datasets.CIFAR10PoissonNoise,
                                    noise_lam=args.noise_lam,
                                    noise_lam_high=args.noise_lam_high,
                                    clip=args.clip,
                                    seed=args.data_seed)
    elif args.dataset == 'CIFAR10PG':
        Dataset = functools.partial(datasets.CIFAR10PoissonGaussianNoise,
                                    noise_lam=args.noise_lam,
                                    noise_scale=args.noise_scale,
                                    noise_lam_high=args.noise_lam_high,
                                    noise_scale_high=args.noise_scale_high,
                                    clip=args.clip,
                                    seed=args.data_seed)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    dataset = Dataset(root=args.dataroot,
                      train=True,
                      download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(normalize)
                      ]))

    iterator = util.InfDataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    # Set up output
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Set up models
    if args.model_seed >= 0:
        torch.manual_seed(args.model_seed)
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    g_params = {
        'latent_dim': args.g_latent_dim,
        'image_size': args.g_image_size,
        'image_channels': args.g_image_channels,
        'channels': args.g_channels,
        'residual_factor': args.g_residual_factor
    }
    netG = net.Generator(**g_params)
    util.save_params(g_params, os.path.join(args.out, 'netG_params.pkl'))
    netG.to(device)
    netG.apply(common.weights_init)
    util.print_network(netG, 'G', os.path.join(args.out, 'netG_arch.txt'))
    if args.g_model_average:
        netG_test = copy.deepcopy(netG)
    else:
        netG_test = netG

    if args.blurvh:
        args.d_image_channels = args.g_image_channels * 2
    else:
        args.d_image_channels = args.g_image_channels
    d_params = {
        'image_channels': args.d_image_channels,
        'channels': args.d_channels,
        'residual_factor': args.d_residual_factor,
        'pooling': args.d_pooling
    }
    netD = net.Discriminator(**d_params)
    util.save_params(d_params, os.path.join(args.out, 'netD_params.pkl'))
    netD.to(device)
    netD.apply(common.weights_init)
    util.print_network(netD, 'D', os.path.join(args.out, 'netD_arch.txt'))

    if args.gn_train:
        if args.implicit:
            args.gn_latent_dim = args.g_latent_dim * 2
        else:
            args.gn_latent_dim = args.g_latent_dim
        gn_params = {
            'latent_dim': args.gn_latent_dim,
            'image_size': args.g_image_size,
            'image_channels': args.g_image_channels,
            'channels': args.g_channels,
            'residual_factor': args.g_residual_factor
        }
        netGn = net.Generator(**gn_params)
        util.save_params(gn_params, os.path.join(args.out, 'netGn_params.pkl'))
        netGn.to(device)
        netGn.apply(common.weights_init)
        util.print_network(netGn, 'Gn', os.path.join(args.out,
                                                     'netGn_arch.txt'))
        if args.g_model_average:
            netGn_test = copy.deepcopy(netGn)
        else:
            netGn_test = netGn
    else:
        netGn, netGn_test = None, None

    # Set up measure
    if args.noise_measure:
        if args.dataset == 'CIFAR10':
            noise_measure = None
        elif args.dataset == 'CIFAR10AG':
            noise_measure = functools.partial(
                measure.additive_gaussian_noise_measure,
                noise_scale=args.noise_scale,
                noise_scale_high=args.noise_scale_high,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10LG':
            noise_measure = functools.partial(
                measure.local_gaussian_noise_measure,
                noise_scale=args.noise_scale,
                patch_size=args.patch_size,
                noise_scale_high=args.noise_scale_high,
                patch_max_size=args.patch_max_size,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10U':
            noise_measure = functools.partial(
                measure.uniform_noise_measure,
                noise_scale=args.noise_scale,
                noise_scale_high=args.noise_scale_high,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10MIX':
            noise_measure = functools.partial(
                measure.mixture_noise_measure,
                noise_scale_list=args.noise_scale_list,
                mixture_rate_list=args.mixture_rate_list,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10BG':
            noise_measure = functools.partial(
                measure.brown_gaussian_noise_measure,
                noise_scale=args.noise_scale,
                noise_scale_high=args.noise_scale_high,
                kernel_size=args.kernel_size,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10ABG':
            noise_measure = functools.partial(
                measure.additive_brown_gaussian_noise_measure,
                noise_scale=args.noise_scale,
                noise_scale_high=args.noise_scale_high,
                kernel_size=args.kernel_size,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10MG':
            noise_measure = functools.partial(
                measure.multiplicative_gaussian_noise_measure,
                multi_noise_scale=args.multi_noise_scale,
                multi_noise_scale_high=args.multi_noise_scale_high,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10AMG':
            noise_measure = functools.partial(
                measure.additive_multiplicative_gaussian_noise_measure,
                noise_scale=args.noise_scale,
                multi_noise_scale=args.multi_noise_scale,
                noise_scale_high=args.noise_scale_high,
                multi_noise_scale_high=args.multi_noise_scale_high,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10P':
            noise_measure = functools.partial(
                measure.poisson_noise_measure,
                noise_lam=args.noise_lam,
                noise_lam_high=args.noise_lam_high,
                image_range=args.image_range)
        elif args.dataset == 'CIFAR10PG':
            noise_measure = functools.partial(
                measure.poisson_gaussian_noise_measure,
                noise_lam=args.noise_lam,
                noise_scale=args.noise_scale,
                noise_lam_high=args.noise_lam_high,
                noise_scale_high=args.noise_scale_high,
                image_range=args.image_range)
    else:
        noise_measure = None

    # Set up optimziers
    optimizerG = optim.Adam(netG.parameters(),
                            lr=args.g_lr,
                            betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netD.parameters(),
                            lr=args.d_lr,
                            betas=(args.beta1, args.beta2))
    if args.gn_train:
        optimizerGn = optim.Adam(netGn.parameters(),
                                 lr=args.g_lr,
                                 betas=(args.beta1, args.beta2))
    else:
        optimizerGn = None

    # Set up learning rate schedulers
    def lr_lambda(iteration):
        if args.num_iterations_decay > 0:
            lr = 1.0 - max(0,
                           (iteration + 1 -
                            (args.num_iterations - args.num_iterations_decay)
                            )) / float(args.num_iterations_decay)
        else:
            lr = 1.0
        return lr

    lr_schedulers = []
    lr_schedulers.append(
        optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr_lambda))
    lr_schedulers.append(
        optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda))
    if args.gn_train:
        lr_schedulers.append(
            optim.lr_scheduler.LambdaLR(optimizerGn, lr_lambda=lr_lambda))

    # Set up trainer
    trainter_params = {
        'iterator': iterator,
        'models': (netG, netD, netGn),
        'models_test': (netG_test, netGn_test),
        'measures': (noise_measure, ),
        'optimizers': (optimizerG, optimizerD, optimizerGn),
        'lr_schedulers': lr_schedulers,
        'batch_size': args.batch_size,
        'g_bs_multiple': args.g_bs_multiple,
        'num_critic': args.num_critic,
        'lambdas': (args.lambda_r1, args.lambda_ds),
        'model_averages': (args.g_model_average, args.g_model_average),
        'model_average_beta': args.model_average_beta,
        'image_range': args.image_range,
        'implicit': args.implicit,
        'prior': args.prior,
        'rotation': args.rotation,
        'channel_shuffle': args.channel_shuffle,
        'color_inversion': args.color_inversion,
        'blurvh': args.blurvh,
        'clip_range': args.clip_range,
        'device': device
    }
    trainer = Trainer(**trainter_params)

    # Set up visualizer and logger
    visualizer = Visualizer(netG_test,
                            netGn_test,
                            device,
                            args.out,
                            args.implicit,
                            args.prior,
                            args.rotation,
                            args.channel_shuffle,
                            args.color_inversion,
                            args.num_columns,
                            image_range=args.image_range)
    logger = Logger(args.out, 'loss')

    # Print args
    util.print_args(args, os.path.join(args.out, 'args.txt'))

    # Train
    while trainer.iteration < args.num_iterations:
        iter_start_time = time.time()
        trainer.update()

        if (args.display_interval > 0
                and trainer.iteration % args.display_interval == 0):
            t = (time.time() - iter_start_time) / args.batch_size
            logger.log(trainer.iteration, trainer.get_current_loss(), t)

        if (args.snapshot_interval > 0
                and trainer.iteration % args.snapshot_interval == 0):
            torch.save(
                netG_test.state_dict(),
                os.path.join(args.out, 'netG_iter_%d.pth' % trainer.iteration))
            torch.save(
                netD.state_dict(),
                os.path.join(args.out, 'netD_iter_%d.pth' % trainer.iteration))
            if args.gn_train:
                torch.save(
                    netGn_test.state_dict(),
                    os.path.join(args.out,
                                 'netGn_iter_%d.pth' % trainer.iteration))

        if (args.visualize_interval > 0
                and trainer.iteration % args.visualize_interval == 0):
            visualizer.visualize(trainer.iteration)


if __name__ == '__main__':
    main()
