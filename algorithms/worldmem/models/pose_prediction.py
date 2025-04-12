import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePredictionNet(nn.Module):
    def __init__(self, img_channels=16, img_feat_dim=256, pose_dim=5, action_dim=25, hidden_dim=128):
        super(PosePredictionNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        self.fc_img = nn.Linear(128, img_feat_dim)  
        
        self.mlp_motion = nn.Sequential(
            nn.Linear(pose_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(img_feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim)
        )

    def forward(self, img, action, pose):
        img_feat = self.cnn(img).view(img.size(0), -1)
        img_feat = self.fc_img(img_feat)
        
        motion_feat = self.mlp_motion(torch.cat([pose, action], dim=1))
        fused_feat = torch.cat([img_feat, motion_feat], dim=1)
        pose_next_pred = self.fc_out(fused_feat)
        
        return pose_next_pred