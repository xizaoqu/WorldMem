import gradio as gr
import time

import sys
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import subprocess
from PIL import Image
from datetime import datetime
import spaces
from algorithms.worldmem import WorldMemMinecraft
from huggingface_hub import hf_hub_download
import tempfile
import os
import requests

torch.set_float32_matmul_precision("high")

def load_custom_checkpoint(algo, checkpoint_path):
    try:
        hf_ckpt = str(checkpoint_path).split('/')
        repo_id = '/'.join(hf_ckpt[:2])
        file_name = '/'.join(hf_ckpt[2:])
        model_path = hf_hub_download(repo_id=repo_id, 
                            filename=file_name)
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        algo.load_state_dict(ckpt['state_dict'], strict=False)
        print("Load: ", model_path)
    except:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        algo.load_state_dict(ckpt['state_dict'], strict=False)      
        print("Load: ", checkpoint_path)  

def download_assets_if_needed():
    ASSETS_URL_BASE = "https://huggingface.co/spaces/yslan/worldmem/resolve/main/assets/examples"
    ASSETS_DIR = "assets/examples"
    ASSETS = ['case1.npz', 'case2.npz', 'case3.npz', 'case4.npz']

    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    
    # Download assets if they don't exist (total 4 files)
    for filename in ASSETS:
        filepath = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            url = f"{ASSETS_URL_BASE}/{filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {filename}: {response.status_code}")

def parse_input_to_tensor(input_str):
    """
    Convert an input string into a (sequence_length, 25) tensor, where each row is a one-hot representation 
    of the corresponding action key.

    Args:
        input_str (str): A string consisting of "WASD" characters (e.g., "WASDWS").

    Returns:
        torch.Tensor: A tensor of shape (sequence_length, 25), where each row is a one-hot encoded action.
    """
    # Get the length of the input sequence
    seq_len = len(input_str)
    
    # Initialize a zero tensor of shape (seq_len, 25)
    action_tensor = torch.zeros((seq_len, 25))

    # Iterate through the input string and update the corresponding positions
    for i, char in enumerate(input_str):
        action, value = KEY_TO_ACTION.get(char.upper())  # Convert to uppercase to handle case insensitivity
        if action and action in ACTION_KEYS:
            index = ACTION_KEYS.index(action)
            action_tensor[i, index] = value  # Set the corresponding action index to 1

    return action_tensor

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Load an image and convert it to a 0-1 normalized tensor.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        torch.Tensor: Image tensor of shape (C, H, W), normalized to [0,1].
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    else:
        image = image_path
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and normalizes to [0,1]
    ])
    return transform(image)

def enable_amp(model, precision="16-mixed"):
    original_forward = model.forward

    def amp_forward(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16 if precision == "16-mixed" else torch.bfloat16):
            return original_forward(*args, **kwargs)

    model.forward = amp_forward
    return model

download_assets_if_needed()

ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraY",
    "cameraX",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

# Mapping of input keys to action names
KEY_TO_ACTION = {
    "Q": ("forward", 1),
    "E": ("back", 1),    
    "W": ("cameraY", -1),
    "S": ("cameraY", 1),
    "A": ("cameraX", -1),
    "D": ("cameraX", 1),
    "U": ("drop", 1),
    "N": ("noop", 1),
    "1": ("hotbar.1", 1),
}

example_images = [
    ["1", "assets/ice_plains.png", "turn rightgo backward→look up→turn left→look down→turn right→go forward→turn left", 20, 3, 8],
    ["2", "assets/place.png", "put item→go backward→put item→go backward→go around", 20, 3, 8],
    ["3", "assets/rain_sunflower_plains.png", "turn right→look up→turn right→look down→turn left→go backward→turn left", 20, 3, 8],
    ["4", "assets/desert.png", "turn 360 degree→turn right→go forward→turn left", 20, 3, 8],
]

memory_frames = []
input_history = ""
ICE_PLAINS_IMAGE = "assets/ice_plains.png"
DESERT_IMAGE = "assets/desert.png"
SAVANNA_IMAGE = "assets/savanna.png"
PLAINS_IMAGE = "assets/plans.png"
PLACE_IMAGE = "assets/place.png"
SUNFLOWERS_IMAGE = "assets/sunflower_plains.png"
SUNFLOWERS_RAIN_IMAGE = "assets/rain_sunflower_plains.png"

device = torch.device('cuda')

def save_video(frames, path="output.mp4", fps=10):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-crf", "23", "-preset", "medium", path
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path

cfg = OmegaConf.load("configurations/huggingface.yaml")
worldmem = WorldMemMinecraft(cfg)
load_custom_checkpoint(algo=worldmem.diffusion_model, checkpoint_path=cfg.diffusion_path)
load_custom_checkpoint(algo=worldmem.vae, checkpoint_path=cfg.vae_path)
load_custom_checkpoint(algo=worldmem.pose_prediction_model, checkpoint_path=cfg.pose_predictor_path)
worldmem.to("cuda").eval()
# worldmem = enable_amp(worldmem, precision="16-mixed")

actions = np.zeros((1, 25), dtype=np.float32)
poses = np.zeros((1, 5), dtype=np.float32)



def get_duration_single_image_to_long_video(first_frame, action, first_pose, device, self_frames, self_actions, 
                            self_poses, self_memory_c2w, self_frame_idx):
    return 5 * len(action) if self_actions is not None else 5

@spaces.GPU(duration=get_duration_single_image_to_long_video)
def run_interactive(first_frame, action, first_pose, device, self_frames, self_actions, 
                            self_poses, self_memory_c2w, self_frame_idx):
    new_frame, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx = worldmem.interactive(first_frame,
                                    action,
                                    first_pose, 
                                    device=device,
                                    self_frames=self_frames,
                                    self_actions=self_actions,
                                    self_poses=self_poses,
                                    self_memory_c2w=self_memory_c2w,
                                    self_frame_idx=self_frame_idx)

    return new_frame, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx

def set_denoising_steps(denoising_steps, sampling_timesteps_state):
    worldmem.sampling_timesteps = denoising_steps
    worldmem.diffusion_model.sampling_timesteps = denoising_steps
    sampling_timesteps_state = denoising_steps
    print("set denoising steps to", worldmem.sampling_timesteps)
    return sampling_timesteps_state

def set_context_length(context_length, sampling_context_length_state):
    worldmem.n_tokens = context_length
    sampling_context_length_state = context_length
    print("set context length to", worldmem.n_tokens)
    return sampling_context_length_state

def set_memory_length(memory_length, sampling_memory_length_state):
    worldmem.condition_similar_length = memory_length
    sampling_memory_length_state = memory_length
    print("set memory length to", worldmem.condition_similar_length)
    return sampling_memory_length_state

def set_next_frame_length(next_frame_length, sampling_next_frame_length_state):
    worldmem.next_frame_length = next_frame_length
    sampling_next_frame_length_state = next_frame_length
    print("set next frame length to", worldmem.next_frame_length)
    return sampling_next_frame_length_state

def generate(keys, input_history, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx):
    input_actions = parse_input_to_tensor(keys)

    if self_frames is None:
        new_frame, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx = run_interactive(memory_frames[0],
                                    actions[0],
                                    poses[0],
                                    device=device,
                                    self_frames=self_frames,
                                    self_actions=self_actions,
                                    self_poses=self_poses,
                                    self_memory_c2w=self_memory_c2w,
                                    self_frame_idx=self_frame_idx)

    new_frame, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx = run_interactive(memory_frames[0],
                                    input_actions,
                                    None,
                                    device=device,
                                    self_frames=self_frames,
                                    self_actions=self_actions,
                                    self_poses=self_poses,
                                    self_memory_c2w=self_memory_c2w,
                                    self_frame_idx=self_frame_idx)

    memory_frames = np.concatenate([memory_frames, new_frame[:,0]])


    out_video = memory_frames.transpose(0,2,3,1).copy()
    out_video = np.clip(out_video, a_min=0.0, a_max=1.0)
    out_video = (out_video * 255).astype(np.uint8)

    last_frame = out_video[-1].copy()
    border_thickness = 2
    out_video[-len(new_frame):, :border_thickness, :, :] = [255, 0, 0]
    out_video[-len(new_frame):, -border_thickness:, :, :] = [255, 0, 0]
    out_video[-len(new_frame):, :, :border_thickness, :] = [255, 0, 0]
    out_video[-len(new_frame):, :, -border_thickness:, :] = [255, 0, 0]

    temporal_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    save_video(out_video, temporal_video_path)
    input_history += keys

    
    # now = datetime.now()
    # folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    # folder_path = os.path.join("/mnt/xiaozeqi/worldmem/output_material", folder_name)
    # os.makedirs(folder_path, exist_ok=True)
    # data_dict = {
    #     "input_history": input_history,
    #     "memory_frames": memory_frames,
    #     "self_frames": self_frames,
    #     "self_actions": self_actions,
    #     "self_poses": self_poses,
    #     "self_memory_c2w": self_memory_c2w,
    #     "self_frame_idx": self_frame_idx,
    # }

    # np.savez(os.path.join(folder_path, "data_bundle.npz"), **data_dict)

    return last_frame, temporal_video_path, input_history, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx

def reset(selected_image):
    self_frames = None
    self_poses = None
    self_actions = None
    self_memory_c2w = None
    self_frame_idx = None
    memory_frames = load_image_as_tensor(selected_image).numpy()[None]
    input_history = ""

    new_frame, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx = run_interactive(memory_frames[0],
                                actions[0],
                                poses[0],
                                device=device,
                                self_frames=self_frames,
                                self_actions=self_actions,
                                self_poses=self_poses,
                                self_memory_c2w=self_memory_c2w,
                                self_frame_idx=self_frame_idx,
                                )

    return input_history, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx

def on_image_click(selected_image):
    input_history, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx = reset(selected_image)
    return input_history, selected_image, selected_image, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx

def set_memory(examples_case):
    if examples_case == '1':
        data_bundle = np.load("assets/examples/case1.npz")
        input_history = data_bundle['input_history'].item()
        memory_frames = data_bundle['memory_frames']
        self_frames = data_bundle['self_frames']
        self_actions = data_bundle['self_actions']
        self_poses = data_bundle['self_poses']
        self_memory_c2w = data_bundle['self_memory_c2w']
        self_frame_idx = data_bundle['self_frame_idx']
    elif examples_case == '2':
        data_bundle = np.load("assets/examples/case2.npz")
        input_history = data_bundle['input_history'].item()
        memory_frames = data_bundle['memory_frames']
        self_frames = data_bundle['self_frames']
        self_actions = data_bundle['self_actions']
        self_poses = data_bundle['self_poses']
        self_memory_c2w = data_bundle['self_memory_c2w']
        self_frame_idx = data_bundle['self_frame_idx']
    elif examples_case == '3':
        data_bundle = np.load("assets/examples/case3.npz")
        input_history = data_bundle['input_history'].item()
        memory_frames = data_bundle['memory_frames']
        self_frames = data_bundle['self_frames']
        self_actions = data_bundle['self_actions']
        self_poses = data_bundle['self_poses']
        self_memory_c2w = data_bundle['self_memory_c2w']
        self_frame_idx = data_bundle['self_frame_idx']
    elif examples_case == '4':
        data_bundle = np.load("assets/examples/case4.npz")
        input_history = data_bundle['input_history'].item()
        memory_frames = data_bundle['memory_frames']
        self_frames = data_bundle['self_frames']
        self_actions = data_bundle['self_actions']
        self_poses = data_bundle['self_poses']
        self_memory_c2w = data_bundle['self_memory_c2w']
        self_frame_idx = data_bundle['self_frame_idx']

    out_video = memory_frames.transpose(0,2,3,1)
    out_video = np.clip(out_video, a_min=0.0, a_max=1.0)
    out_video = (out_video * 255).astype(np.uint8)

    temporal_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    save_video(out_video, temporal_video_path)

    return input_history, out_video[-1], temporal_video_path, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx

css = """
h1 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # WORLDMEM: Long-term Consistent World Simulation with Memory
        """
        )

    gr.Markdown(
        """
        ## 🚀 How to Explore WorldMem

        Follow these simple steps to get started:

        1. **Choose a scene**.
        2. **Input your action sequence**.
        3. **Click "Generate"**.

        - You can continuously click **"Generate"** to **extend the video** and observe how well the world maintains consistency over time.
        - For best performance, we recommend **running locally** (1s/frame on H100) instead of Spaces (5s/frame).
        - ⭐️ If you like this project, please [give it a star on GitHub]()!
        - 💬 For questions or feedback, feel free to open an issue or email me at **zeqixiao1@gmail.com**.

        Happy exploring! 🌍
        """
    )
        # <div style="text-align: center;">
        # <!-- Public Website -->
        # <a style="display:inline-block" href="https://nirvanalan.github.io/projects/GA/">
        #     <img src="https://img.shields.io/badge/public_website-8A2BE2">
        # </a>

        # <!-- GitHub Stars -->
        # <a style="display:inline-block; margin-left: .5em" href="https://github.com/NIRVANALAN/GaussianAnything">
        #     <img src="https://img.shields.io/github/stars/NIRVANALAN/GaussianAnything?style=social">
        # </a>

        # <!-- Project Page -->
        # <a style="display:inline-block; margin-left: .5em" href="https://nirvanalan.github.io/projects/GA/">
        #     <img src="https://img.shields.io/badge/project_page-blue">
        # </a>

        # <!-- arXiv Paper -->
        # <a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/XXXX.XXXXX">
        #     <img src="https://img.shields.io/badge/arXiv-paper-red">
        # </a>
        # </div>

    example_actions = {"turn left→turn right": "AAAAAAAAAAAADDDDDDDDDDDD", 
                        "turn 360 degree": "AAAAAAAAAAAAAAAAAAAAAAAA", 
                        "turn right→go backward→look up→turn left→look down": "DDDDDDDDEEEEEEEEEESSSAAAAAAAAWWW", 
                        "turn right→go forward→turn right": "DDDDDDDDDDDDQQQQQQQQQQQQQQQDDDDDDDDDDDD", 
                        "turn right→look up→turn right→look down": "DDDDWWWDDDDDDDDDDDDDDDDDDDDSSS", 
                        "put item→go backward→put item→go backward":"SSUNNWWEEEEEEEEEAAASSUNNWWEEEEEEEEE"}

    selected_image = gr.State(ICE_PLAINS_IMAGE)

    with gr.Row(variant="panel"):
        with gr.Column():
            gr.Markdown("🖼️ Start from this frame.")
            image_display = gr.Image(value=selected_image.value, interactive=False, label="Current Frame")
        with gr.Column():
            gr.Markdown("🎞️ Generated videos. New contents are marked in red box.")
            video_display = gr.Video(autoplay=True, loop=True)

    gr.Markdown("### 🏞️ Choose a scene and start generation.")

    with gr.Row():
        image_display_1 = gr.Image(value=SUNFLOWERS_IMAGE, interactive=False, label="Sunflower Plains")
        image_display_2 = gr.Image(value=DESERT_IMAGE, interactive=False, label="Desert")
        image_display_3 = gr.Image(value=SAVANNA_IMAGE, interactive=False, label="Savanna")
        image_display_4 = gr.Image(value=ICE_PLAINS_IMAGE, interactive=False, label="Ice Plains")
        image_display_5 = gr.Image(value=SUNFLOWERS_RAIN_IMAGE, interactive=False, label="Rainy Sunflower Plains")
        image_display_6 = gr.Image(value=PLACE_IMAGE, interactive=False, label="Place")        


    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            gr.Markdown("### 🕹️ Input action sequences for interaction.")
            input_box = gr.Textbox(label="Action Sequences", placeholder="Enter action sequences here, e.g. (AAAAAAAAAAAADDDDDDDDDDDD)", lines=1, max_lines=1)
            log_output = gr.Textbox(label="History Sequences", interactive=False)
            gr.Markdown(
                """
                ### 💡 Action Key Guide

                <pre style="font-family: monospace; font-size: 14px; line-height: 1.6;">
                W: Turn up      S: Turn down     A: Turn left     D: Turn right
                Q: Go forward   E: Go backward   N: No-op         U: Use item
                </pre>
                """
            )
            gr.Markdown("### 👇 Click to quickly set action sequence examples.")
            with gr.Row():
                buttons = []
                for action_key in list(example_actions.keys())[:2]:
                    with gr.Column(scale=len(action_key)):
                        buttons.append(gr.Button(action_key))
            with gr.Row():
                for action_key in list(example_actions.keys())[2:4]:
                    with gr.Column(scale=len(action_key)):
                        buttons.append(gr.Button(action_key))
            with gr.Row():
                for action_key in list(example_actions.keys())[4:6]:
                    with gr.Column(scale=len(action_key)):
                        buttons.append(gr.Button(action_key))

        with gr.Column(scale=1):
            submit_button = gr.Button("🎬 Generate!", variant="primary")
            reset_btn = gr.Button("🔄 Reset")

            # gr.Markdown("<div style='flex-grow:1; height: 100px'></div>")

            gr.Markdown("### ⚙️ Advanced Settings")

            slider_denoising_step = gr.Slider(
                minimum=10, maximum=50, value=worldmem.sampling_timesteps, step=1,
                label="Denoising Steps",
                info="Higher values yield better quality but slower speed"
            )
            slider_context_length = gr.Slider(
                minimum=2, maximum=10, value=worldmem.n_tokens, step=1,
                label="Context Length",
                info="How many previous frames in temporal context window."
            )
            slider_memory_length = gr.Slider(
                minimum=4, maximum=16, value=worldmem.condition_similar_length, step=1,
                label="Memory Length",
                info="How many previous frames in memory window."
            )
            slider_next_frame_length = gr.Slider(
                minimum=1, maximum=5, value=worldmem.next_frame_length, step=1,
                label="Next Frame Length",
                info="How many next frames to generate at once."
            )
    
    sampling_timesteps_state = gr.State(worldmem.sampling_timesteps)
    sampling_context_length_state = gr.State(worldmem.n_tokens)
    sampling_memory_length_state = gr.State(worldmem.condition_similar_length)
    sampling_next_frame_length_state = gr.State(worldmem.next_frame_length)

    memory_frames = gr.State(load_image_as_tensor(selected_image.value)[None].numpy())
    self_frames = gr.State()
    self_actions = gr.State()
    self_poses = gr.State()
    self_memory_c2w = gr.State()
    self_frame_idx = gr.State()

    def set_action(action):
        return action
    


    for button, action_key in zip(buttons, list(example_actions.keys())):
            button.click(set_action, inputs=[gr.State(value=example_actions[action_key])], outputs=input_box)

    gr.Markdown("### 👇 Click to review generated examples, and continue generation based on them.")

    example_case = gr.Textbox(label="Case", visible=False)
    image_output = gr.Image(visible=False) 

    examples = gr.Examples(
        examples=example_images,
        inputs=[example_case, image_output, log_output, slider_denoising_step, slider_context_length, slider_memory_length],
        cache_examples=False
    )

    example_case.change(
        fn=set_memory,
        inputs=[example_case],
        outputs=[log_output, image_display, video_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx]
    )

    submit_button.click(generate, inputs=[input_box, log_output, memory_frames, 
                                          self_frames, self_actions, self_poses, 
                                          self_memory_c2w, self_frame_idx], 
                                          outputs=[image_display, video_display, log_output, 
                                                                                                                                                                  memory_frames, self_frames, self_actions, self_poses, 
                                                                                                                                                                  self_memory_c2w, self_frame_idx])
    reset_btn.click(reset, inputs=[selected_image], outputs=[log_output, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])
    image_display_1.select(lambda: on_image_click(SUNFLOWERS_IMAGE), outputs=[log_output, selected_image, image_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])
    image_display_2.select(lambda: on_image_click(DESERT_IMAGE), outputs=[log_output, selected_image, image_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])
    image_display_3.select(lambda: on_image_click(SAVANNA_IMAGE), outputs=[log_output, selected_image, image_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])
    image_display_4.select(lambda: on_image_click(ICE_PLAINS_IMAGE), outputs=[log_output, selected_image, image_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])
    image_display_5.select(lambda: on_image_click(SUNFLOWERS_RAIN_IMAGE), outputs=[log_output, selected_image, image_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])
    image_display_6.select(lambda: on_image_click(PLACE_IMAGE), outputs=[log_output, selected_image,image_display, memory_frames, self_frames, self_actions, self_poses, self_memory_c2w, self_frame_idx])

    slider_denoising_step.change(fn=set_denoising_steps, inputs=[slider_denoising_step, sampling_timesteps_state], outputs=sampling_timesteps_state)
    slider_context_length.change(fn=set_context_length, inputs=[slider_context_length, sampling_context_length_state], outputs=sampling_context_length_state)
    slider_memory_length.change(fn=set_memory_length, inputs=[slider_memory_length, sampling_memory_length_state], outputs=sampling_memory_length_state)
    slider_next_frame_length.change(fn=set_next_frame_length, inputs=[slider_next_frame_length, sampling_next_frame_length_state], outputs=sampling_next_frame_length_state)

demo.launch()
