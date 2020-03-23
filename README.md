# NR-GAN: Noise Robust Generative Adversarial Networks (CVPR 2020)

This repository provides PyTorch implementation for **noise robust GAN** ([**NR-GAN**](https://arxiv.org/abs/1911.11776)).
NR-GAN is unique in that it can learn a *clean image generator* even when only *noisy images* are available for training.

<img src="docs/images/examples.png" width=100% alt="NR-GAN examples">

**NOTE:**
In our previous studies, we also proposed GANs for *label noise*.
Please check them from the links below.

- **Label-noise robust GAN (rGAN)**: [[Paper]](https://arxiv.org/abs/1811.11165) [[Project]](https://takuhirok.github.io/rGAN/) [[Code]](https://github.com/takuhirok/rGAN/)
- **Classifier's posterior GAN (CP-GAN)**: [[Paper]](https://arxiv.org/abs/1811.11163) [[Project]](https://takuhirok.github.io/CP-GAN/) [[Code]](https://github.com/takuhirok/CP-GAN/)

## Paper

Noise Robust Generative Adversarial Networks.<br>
[Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/) and [Tatsuya Harada](https://www.mi.t.u-tokyo.ac.jp/harada/).<br>
In CVPR 2020.

[[Paper]](https://arxiv.org/abs/1911.11776)
[[Project]](https://takuhirok.github.io/NR-GAN/)

## Code
The code will be released soon.
Meanwhile, please check our [paper](https://arxiv.org/abs/1911.11776) and [project page](https://takuhirok.github.io/NR-GAN/).

## Citation
If you find this work useful for your research, please cite our paper.

```
@inproceedings{kaneko2020NR-GAN,
  title={Noise Robust Generative Adversarial Networks},
  author={Kaneko, Takuhiro and Harada, Tatsuya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Related work

1. T. Kaneko, Y. Ushiku, T. Harada. Label-Noise Robust Generative Adversarial Networks, In CVPR 2019. [[Paper]](https://arxiv.org/abs/1811.11165) [[Project]](https://takuhirok.github.io/rGAN/) [[Code]](https://github.com/takuhirok/rGAN/)
2. T. Kaneko, Y. Ushiku, T. Harada. Class-Distinct and Class-Mutual Image Generation with GANs, In BMVC 2019. [[Paper]](https://arxiv.org/abs/1811.11163) [[Project]](https://takuhirok.github.io/CP-GAN/) [[Code]](https://github.com/takuhirok/CP-GAN/)