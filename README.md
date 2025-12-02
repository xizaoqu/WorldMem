
<br>
<p align="center">

<p align="center">
  <img src="assets/worldmem_logo.png" alt="WORLDMEM Icon" width="80"/>
</p>
<h1 align="center"><strong>WorldMem: Long-term Consistent World Simulation <br> with Memory</strong></h1>
  <p align="center"><span><a href=""></a></span>
              <a href="https://xizaoqu.github.io">Zeqi Xiao<sup>1</sup></a>
              <a href="https://nirvanalan.github.io/">Yushi Lan<sup>1</sup></a>
              <a href="https://zhouyifan.net/about/">Yifan Zhou<sup>1</sup></a>
              <a href="https://vicky0522.github.io/Wenqi-Ouyang/">Wenqi Ouyang<sup>1</sup></a>
              <a href="https://williamyang1991.github.io/">Shuai Yang<sup>2</sup></a>
              <a href="https://zengyh1900.github.io/">Yanhong Zeng<sup>3</sup></a>
              <a href="https://xingangpan.github.io/">Xingang Pan<sup>1</sup></a>    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University, <br> <sup>2</sup>Wangxuan Institute of Computer Technology, Peking University,<br>  <sup>3</sup>Shanghai AI Laboratory
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
conda install -c conda-forge ffmpeg=4.3.2
```


## Quick start

```
python app.py
```

## Run

To enable cloud logging with [Weights & Biases (wandb)](https://wandb.ai/site), follow these steps:

1. Sign up for a wandb account.
2. Run the following command to log in:

    ```bash
    wandb login
    ```

3. Open `configurations/training.yaml` and set the `entity` and `project` field to your wandb username.

---

### Training

Download pretrained weights from [Oasis](https://github.com/etched-ai/open-oasis).

Training the model on 4 H100 GPUs, it converges after approximately 500K steps.
We observe that gradually increasing task difficulty improves performance. Thus, we adopt a multi-stage training strategy:
, 
```bash
sh train_stage_1.sh   # Small range, no vertical turning
sh train_stage_2.sh   # Large range, no vertical turning
sh train_stage_3.sh   # Large range, with vertical turning
```

To resume training from a previous checkpoint, configure the `resume` and `output_dir` variables in the corresponding `.sh` script.

---

### Inference

To run inference:

```bash
sh infer.sh
```

You can either **load the diffusion model and VAE separately**:

```bash
+diffusion_model_path=zeqixiao/worldmem_checkpoints/diffusion_only.ckpt \
+vae_path=zeqixiao/worldmem_checkpoints/vae_only.ckpt \
+customized_load=true \
+seperate_load=true \
```

Or **load a combined checkpoint**:

```bash
+load=your_model_path \
+customized_load=true \
+seperate_load=false \
```

### Evaluation

To run evaluation:

```bash
sh evaluate.sh
```

This script reproduces the results in Table 1 (beyond context window). Evaluating 1 case on 1 A100 GPU takes approximately 6 minutes. You can adjust `experiment.test.limit_batch` to specify the number of cases to evaluate.

Visual results will be saved by default to a timestamped directory (e.g., `outputs/2025-11-30/00-02-42`).

To calculate the FID score, run:

```bash
python calculate_fid.py --videos_dir <path_to_videos>
```

For example:

```bash
python calculate_fid.py --videos_dir outputs/2025-11-30/00-02-42/videos/test_vis
```

**Expected Results:**

| Metric | Value  |
|--------|--------|
| PSNR   | 19.34  |
| LPIPS  | 0.1667 |
| FID    | 15.13  |

*Note: FID is computed over 5000 frames.*

*Note: Previous versions incorrectly used `data_range=2.0` for PSNR calculation, but the decoded video data is in the range [0, 1], so `data_range=1.0` should be used. This bug inflated PSNR values by approximately 6 dB. We have now corrected this in the latest version by clipping predictions to [0, 1] before metric computation. The relative performance comparisons and conclusions remain unchanged.*

---

## Dataset

Download the Minecraft dataset from [Hugging Face](https://huggingface.co/datasets/zeqixiao/worldmem_minecraft_dataset)

Place the dataset in the following directory structure:

```
data/
└── minecraft/
    ├── training/
    └── validation/
    └── test/
```

## Data Generation

After setting up the environment as described in [MineDojo's GitHub repository](https://github.com/MineDojo/MineDojo), you can generate data using the following command:

```bash
xvfb-run -a python data_generator.py -o data/test -z 4 --env_type plains
```

**Parameters:**
- `-o`: Output directory for generated data
- `-z`: Number of parallel workers
- `--env_type`: Environment type (e.g., `plains`, `forest`, `desert`)


## TODO

- [x] Release inference models and weights;
- [x] Release training pipeline on Minecraft;
- [x] Release training data on Minecraft;
- [x] Release evaluation scripts and data generator.



## 🔗 Citation

If you find our work helpful, please cite:

```
@misc{xiao2025worldmemlongtermconsistentworld,
      title={WORLDMEM: Long-term Consistent World Simulation with Memory}, 
      author={Zeqi Xiao and Yushi Lan and Yifan Zhou and Wenqi Ouyang and Shuai Yang and Yanhong Zeng and Xingang Pan},
      year={2025},
      eprint={2504.12369},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.12369}, 
}
```

## 👏 Acknowledgements
- [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing): Diffusion Forcing provides flexible training and inference strategies for our methods.
- [Minedojo](https://github.com/MineDojo/MineDojo): We collect our Minecraft dataset from Minedojo.
- [Open-oasis](https://github.com/etched-ai/open-oasis): Our model architecture is based on Open-oasis. We also use pretrained VAE and DiT weight from it.
