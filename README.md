
<br>
<p align="center">
<h1 align="center"><strong>WORLDMEM: Long-term Consistent World Generation with Memory</strong></h1>
  <p align="center"><span><a href="https://natanielruiz.github.io/"></a></span>
              <a href="https://github.com/xizaoqu">Zeqi Xiao<sup>1</sup></a>
              <a href="https://nirvanalan.github.io/">Yushi Lan<sup>1</sup></a>
              <a href="https://zhouyifan.net/about/">Yifan Zhou<sup>1</sup></a>
              <a href="https://vicky0522.github.io/Wenqi-Ouyang/">Wenqi Ouyang<sup>1</sup></a>
              <a href="https://williamyang1991.github.io/">Shuai Yang<sup>2</sup></a>
              <a href="https://zengyh1900.github.io/">Yanhong Zeng<sup>3</sup></a>
              <a href="https://xingangpan.github.io/">Xingang Pan<sup>1</sup></a>    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University, <br> <sup>2</sup>Wangxuan Institute of Computer Technology, Peking University,<br>  <sup>3</sup>Shanghai AI Laboratry
    </p>
</p>

<p align="center">
  <!-- <a href="https://arxiv.org/abs/2405.14864" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2308.16911-blue?">
  </a> -->
  <a href="https://xizaoqu.github.io/worldmem/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
</p>


## Installation

```
conda create python=3.10 -n worldmem
conda activate worldmem
pip install -r requirements.txt
```


## Quick start



## TODO

- [x] Release inference models and weight;
- [] Release training pipeline on MineCraft;
- [] Release training data on MineCraft;



## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
xiao2025trajectory,
title={Trajectory attention for fine-grained video motion control},
author={Zeqi Xiao and Wenqi Ouyang and Yifan Zhou and Shuai Yang and Lei Yang and Jianlou Si and Xingang Pan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2z1HT5lw5M}
}
```

## 👏 Acknowledgements
- [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing): Diffusion Forcing provides flexible training and inference strategies for our methods.
- [Minedojo](https://github.com/MineDojo/MineDojo): We collect our minecraft dataset from Minedojo.
- [Open-oasis](https://github.com/etched-ai/open-oasis): Our model architecture is based on Open-oasis. We also use pretrained VAE and DiT weight from it.