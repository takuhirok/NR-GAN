import cv2
import numpy as np

import torchvision.datasets as datasets


class CIFAR10Noise(datasets.CIFAR10):
    """CIFAR10 Dataset with noise.

    Args:
        clip (bool): If True, clips a value between 0 and 1 (default: True).
        seed (int): Random seed (default: 0).

    This is a subclass of the `CIFAR10` Dataset.
    """
    def __init__(self, clip=True, seed=0, **kwargs):
        self.clip = clip
        self.seed = seed
        super(CIFAR10Noise, self).__init__(**kwargs)
        assert (seed + 1) * len(self) - 1 <= 2**32 - 1

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        noise = self.generate_noise(index)

        img = img / 255.
        noise = noise / 255.
        img = img + noise

        img, target = self.postprocess(img, target)
        return img, target

    def postprocess(self, img, target):
        if self.clip:
            img = np.clip(img, 0., 1.)

        if self.transform is not None:
            img = img.astype(np.float32)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def generate_noise(self):
        raise NotImplementedError


class CIFAR10AdditiveGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with additive Gaussian noise.

    Args:
        noise_scale (float): The standard deviation of additive Gaussian noise
            (default: 25.).
        noise_scale_high (float): The upper bound of the standard deviation of
            additive Gaussian noise (default: None, i.e., `noise_scale`).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self, noise_scale=25., noise_scale_high=None, **kwargs):
        self.noise_scale = noise_scale
        self.noise_scale_high = noise_scale_high
        super(CIFAR10AdditiveGaussianNoise, self).__init__(**kwargs)

    def generate_noise(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        return rng.randn(*self.data[index].shape) * noise_scale


class CIFAR10LocalGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with local Gaussian noise.

    Args:
        noise_scale (float): The standard deviation of additive Gaussian noise
            (default: 25.).
        patch_size (int): The height/width of the noise patch (default: 16.).
        noise_scale_high (float): The upper bound of the standard deviation of
            additive Gaussian noise (default: None, i.e., `noise_scale`).
        patch_max_size (int): The maximum height/width of the noise patch
            (default: None, i.e., `patch_size`).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 noise_scale=25.,
                 patch_size=16,
                 noise_scale_high=None,
                 patch_max_size=None,
                 **kwargs):
        self.noise_scale = noise_scale
        self.patch_size = patch_size
        self.noise_scale_high = noise_scale_high
        self.patch_max_size = patch_max_size
        super(CIFAR10LocalGaussianNoise, self).__init__(**kwargs)

    def generate_noise(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        patch_shape = (self.data[index].shape[0], self.data[index].shape[1], 1)
        patch = np.zeros(patch_shape, dtype=np.uint8)
        if self.patch_max_size is None:
            patch_width = self.patch_size
            patch_height = self.patch_size
        else:
            patch_width = rng.randint(self.patch_size, self.patch_max_size + 1)
            patch_height = rng.randint(self.patch_size,
                                       self.patch_max_size + 1)
        x = rng.randint(0, patch_shape[1] - patch_width + 1)
        y = rng.randint(0, patch_shape[0] - patch_height + 1)
        patch[y:y + patch_height, x:x + patch_width] = 1
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        noise = rng.randn(*self.data[index].shape) * noise_scale
        return noise * patch


class CIFAR10UniformNoise(CIFAR10Noise):
    """CIFAR10 Dataset with uniform noise.

    Args:
        noise_scale (float): The scale of uniform noise (default: 50.).
        noise_scale_high (float): The upper bound of the scale of uniform noise
            (default: None, i.e., `noise_scale`).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self, noise_scale=50., noise_scale_high=None, **kwargs):
        self.noise_scale = noise_scale
        self.noise_scale_high = noise_scale_high
        super(CIFAR10UniformNoise, self).__init__(**kwargs)

    def generate_noise(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        return rng.uniform(-1, 1, self.data[index].shape) * noise_scale


class CIFAR10MixtureNoise(CIFAR10Noise):
    """CIFAR10 Dataset with mixture noise.

    Args:
        noise_scale_list (float list): The values, except for the last one,
            indicate the standard deviations of additive Gaussian noises. The
            last value indicates the scale of uniform noise (default:
            [15., 25., 50.]).
        mixture_rate_list (float list): The mixture rates of the noises
            (default: [0.7, 0.2, 0.1]).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 noise_scale_list=[15., 25., 50.],
                 mixture_rate_list=[0.7, 0.2, 0.1],
                 **kwargs):
        self.noise_scale_list = noise_scale_list
        self.mixture_rate_list = mixture_rate_list
        super(CIFAR10MixtureNoise, self).__init__(**kwargs)

    def generate_noise(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        height, width, channel = list(self.data[index].shape)
        noise = np.zeros((height * width, channel))
        perm = rng.permutation(height * width)
        rand = rng.rand(height * width)
        cumsum = np.cumsum([0] + self.mixture_rate_list)
        for i, noise_scale in enumerate(self.noise_scale_list):
            inds = (rand >= cumsum[i]) * (rand < cumsum[i + 1])
            if i == len(self.noise_scale_list) - 1:
                noise[perm[inds], :] = rng.uniform(
                    -1, 1, (np.sum(inds), channel)) * noise_scale
            else:
                noise[perm[inds], :] = rng.randn(np.sum(inds),
                                                 channel) * noise_scale
        noise = np.reshape(noise, (height, width, channel))
        return noise


class CIFAR10BrownGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with Brown Gaussian noise.

    Args:
        noise_scale (float): The standard deviation of additive Gaussian noise
            (default: 25.).
        noise_scale_high (float): The upper bound of the standard deviation of
            additive Gaussian noise (default: None, i.e., `noise_scale`).
        kernel_size (int): The Gaussian kernel size (default: 5).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 noise_scale=25.,
                 noise_scale_high=None,
                 kernel_size=5,
                 **kwargs):
        self.noise_scale = noise_scale
        self.noise_scale_high = noise_scale_high
        self.kernel_size = kernel_size
        super(CIFAR10BrownGaussianNoise, self).__init__(**kwargs)
        self.kernel = (cv2.getGaussianKernel(kernel_size, 0) *
                       cv2.getGaussianKernel(kernel_size, 0).transpose())

    def generate_noise(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        noise = rng.randn(*self.data[index].shape) * noise_scale
        return (cv2.GaussianBlur(noise, (self.kernel_size, self.kernel_size),
                                 0,
                                 borderType=cv2.BORDER_CONSTANT) /
                np.sqrt(np.sum(self.kernel**2)))


class CIFAR10AdditiveBrownGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with additive Brown Gaussian noise.

    Args:
        noise_scale (float): The standard deviation of additive Gaussian noise
            (default: 25.).
        noise_scale_high (float): The upper bound of the standard deviation of
            additive Gaussian noise (default: None, i.e., `noise_scale`).
        kernel_size (int): The Gaussian kernel size (default: 5).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 noise_scale=25.,
                 noise_scale_high=None,
                 kernel_size=5,
                 **kwargs):
        self.noise_scale = noise_scale
        self.noise_scale_high = noise_scale_high
        self.kernel_size = kernel_size
        super(CIFAR10AdditiveBrownGaussianNoise, self).__init__(**kwargs)
        self.kernel = (cv2.getGaussianKernel(kernel_size, 0) *
                       cv2.getGaussianKernel(kernel_size, 0).transpose())

    def generate_noise(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        noise = rng.randn(*self.data[index].shape) * noise_scale
        return noise + (cv2.GaussianBlur(noise,
                                         (self.kernel_size, self.kernel_size),
                                         0,
                                         borderType=cv2.BORDER_CONSTANT) /
                        np.sqrt(np.sum(self.kernel**2)))


class CIFAR10MultiplicativeGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with multiplicative Gaussian noise.

    Args:
        multi_noise_scale (float): The standard deviation of multiplicative
            Gaussian noise (default: 25.).
        multi_noise_scale_high (float): The upper bound of the standard
            deviation of multiplicative Gaussian noise (default: None, i.e.,
            `multi_noise_scale`).

    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 multi_noise_scale=25.,
                 multi_noise_scale_high=None,
                 **kwargs):
        self.multi_noise_scale = multi_noise_scale
        self.multi_noise_scale_high = multi_noise_scale_high
        super(CIFAR10MultiplicativeGaussianNoise, self).__init__(**kwargs)

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        img, target = self.data[index], self.targets[index]

        img = img / 255.
        if self.multi_noise_scale_high is None:
            multi_noise_scale = self.multi_noise_scale
        else:
            multi_noise_scale = rng.uniform(self.multi_noise_scale,
                                            self.multi_noise_scale_high)
        noise = rng.randn(*img.shape) * multi_noise_scale * img / 255.
        img = img + noise

        img, target = self.postprocess(img, target)
        return img, target


class CIFAR10AdditiveMultiplicativeGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with additive and multiplicative Gaussian noise.

    Args:
        noise_scale (float): The standard deviation of additive Gaussian noise
            (default: 25.).
        multi_noise_scale (float): The standard deviation of multiplicative
            Gaussian noise (default: 25.).
        noise_scale_high (float): The upper bound of the standard deviation of
            additive Gaussian noise (default: None, i.e., `noise_scale`).
        multi_noise_scale_high (float): The upper bound of the standard
            deviation of multiplicative Gaussian noise (default: None, i.e.,
            `multi_noise_scale`).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 noise_scale=25.,
                 multi_noise_scale=25.,
                 noise_scale_high=None,
                 multi_noise_scale_high=None,
                 **kwargs):
        self.noise_scale = noise_scale
        self.multi_noise_scale = multi_noise_scale
        self.noise_scale_high = noise_scale_high
        self.multi_noise_scale_high = multi_noise_scale_high
        super(CIFAR10AdditiveMultiplicativeGaussianNoise,
              self).__init__(**kwargs)

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        img, target = self.data[index], self.targets[index]

        img = img / 255.
        if self.multi_noise_scale_high is None:
            multi_noise_scale = self.multi_noise_scale
        else:
            multi_noise_scale = rng.uniform(self.multi_noise_scale,
                                            self.multi_noise_scale_high)
        noise = rng.randn(*img.shape) * multi_noise_scale * img / 255.
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        noise = noise + rng.randn(*img.shape) * noise_scale / 255.
        img = img + noise

        img, target = self.postprocess(img, target)
        return img, target


class CIFAR10PoissonNoise(CIFAR10Noise):
    """CIFAR10 Dataset with Poisson noise.

    Args:
        noise_lam (float): The total number of events for Poisson noise
            (default: 30.).
        noise_lam_high (float): The maximum total number of events for Poisson
            noise (default: None, i.e., `noise_lam`).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self, noise_lam=30., noise_lam_high=None, **kwargs):
        self.noise_lam = noise_lam
        self.noise_lam_high = noise_lam_high
        super(CIFAR10PoissonNoise, self).__init__(**kwargs)

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        img, target = self.data[index], self.targets[index]

        img = img / 255.
        if self.noise_lam_high is None:
            noise_lam = self.noise_lam
        else:
            noise_lam = rng.uniform(self.noise_lam, self.noise_lam_high)
        img = rng.poisson(noise_lam * img) / noise_lam

        img, target = self.postprocess(img, target)
        return img, target


class CIFAR10PoissonGaussianNoise(CIFAR10Noise):
    """CIFAR10 Dataset with Poisson-Gaussian noise.

    Args:
        noise_lam (float): The total number of events for Poisson noise
            (default: 30.).
        noise_scale (float): The standard deviation of additive Gaussian noise
            (default: 25.).
        noise_lam_high (float): The maximum total number of events for Poisson
            noise (default: None, i.e., `noise_lam`).
        noise_scale_high (float): The upper bound of the standard deviation of
            additive Gaussian noise (default: None, i.e., `noise_scale`).
        
    This is a subclass of the `CIFAR10Noise` Dataset.
    """
    def __init__(self,
                 noise_lam=30.,
                 noise_scale=3.,
                 noise_lam_high=None,
                 noise_scale_high=None,
                 **kwargs):
        self.noise_lam = noise_lam
        self.noise_lam_high = noise_lam_high
        self.noise_scale = noise_scale
        self.noise_scale_high = noise_scale_high
        super(CIFAR10PoissonGaussianNoise, self).__init__(**kwargs)

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed * len(self) + index)
        img, target = self.data[index], self.targets[index]

        img = img / 255.
        if self.noise_lam_high is None:
            noise_lam = self.noise_lam
        else:
            noise_lam = rng.uniform(self.noise_lam, self.noise_lam_high)
        img = rng.poisson(noise_lam * img) / noise_lam
        if self.noise_scale_high is None:
            noise_scale = self.noise_scale
        else:
            noise_scale = rng.uniform(self.noise_scale, self.noise_scale_high)
        img = img + rng.randn(*img.shape) * noise_scale / 255.

        img, target = self.postprocess(img, target)
        return img, target
