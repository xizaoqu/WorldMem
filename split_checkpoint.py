import torch

ckpt_path = "/mnt/xiaozeqi/diffusionforcing/outputs/2025-03-28/16-45-11/checkpoints/epoch0step595000.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")  # map_location 可根据需要更换

state_dict = checkpoint['state_dict']
pose_prediction_model_dict = {k.replace('pose_prediction_model.', ''): v for k, v in state_dict.items() if k.startswith('pose_prediction_model.')}

torch.save({'state_dict': pose_prediction_model_dict}, "pose_prediction_model_only.ckpt")