import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class VisionEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        import torchvision.models as models

        # Pretrained ResNet18 encoder; we will use the conv features + global pool
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove fc
        self.encoder = nn.Sequential(*modules)
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()

    def forward(self, imgs):
        # imgs: tensor (B,3,H,W) in 0..1
        with torch.no_grad():
            feats = self.encoder(imgs)
            feats = feats.view(feats.size(0), -1)
        return feats


class SimplePolicy(nn.Module):
    def __init__(self, obs_dim=512, n_actions=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.encoder = VisionEncoder(device=device)
        # resnet18 final feature size is 512
        self.policy = SimplePolicy(obs_dim=512, n_actions=6).to(self.device)
        # policy is untrained; use argmax of logits for deterministic action
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def act(self, observation):
        # observation: HxWx3 uint8 RGB
        img = Image.fromarray(observation)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.encoder(x)
            logits = self.policy(feat)
            action = int(torch.argmax(logits, dim=-1).item())
        return action

    def save_policy(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
