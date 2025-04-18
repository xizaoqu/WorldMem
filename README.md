
<br>
<p align="center">

<p align="center">
  <img src="assets/worldmem_logo.png" alt="WORLDMEM Icon" width="80"/>
</p>
<h1 align="center"><strong>WORLDMEM: Long-term Consistent World Simulation with Memory</strong></h1>
  <p align="center"><span><a href=""></a></span>
              <a href="https://xizaoqu.github.io">Zeqi Xiao<sup>1</sup></a>
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
  <a href="https://arxiv.org/abs/2504.12369" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2504.12369-blue?">
  </a>
  <a href="https://xizaoqu.github.io/worldmem/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
<a href="https://huggingface.co/spaces/yslan/worldmem" target="_blank">
  <img src="https://img.shields.io/badge/🤗 HuggingFace-Demo-orange" />
</a>
</p>

https://github.com/user-attachments/assets/fb8a32e2-9470-4819-a93d-c38caf76d72c


## Installation

```
conda create python=3.10 -n worldmem
conda activate worldmem
pip install -r requirements.txt
```


## Quick start

```
python app.py
```

## TODO

- [x] Release inference models and weights;
- [ ] Release training pipeline on Minecraft;
- [ ] Release training data on Minecraft;



## 🔗 Citation

If you find our work helpful, please cite:

```
TBD
```

## 👏 Acknowledgements
- [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing): Diffusion Forcing provides flexible training and inference strategies for our methods.
- [Minedojo](https://github.com/MineDojo/MineDojo): We collect our Minecraft dataset from Minedojo.
- [Open-oasis](https://github.com/etched-ai/open-oasis): Our model architecture is based on Open-oasis. We also use pretrained VAE and DiT weight from it.
