"""Minimal TCC encoder runtime copied from FineProg."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50Conv4cBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(weights=None)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3

    def forward(self, frames):
        frames = self.conv1(frames)
        frames = self.bn1(frames)
        frames = self.relu(frames)
        frames = self.maxpool(frames)
        frames = self.layer1(frames)
        frames = self.layer2(frames)
        return self.layer3(frames)


class TCCTemporalEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv3d_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 512)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 512)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, features):
        batch, clip_len, context_size, channels, height, width = features.shape
        features = features.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * clip_len, channels, context_size, height, width)
        features = self.relu_1(self.conv3d_1(features))
        features = self.relu_2(self.conv3d_2(features))
        features = F.adaptive_max_pool3d(features, output_size=1)
        features = features.reshape(batch * clip_len, 512)
        features = self.relu_fc1(self.fc1(features))
        features = self.relu_fc2(self.fc2(features))
        return self.proj(features).reshape(batch, clip_len, -1)


class TCCEncoder(nn.Module):
    def __init__(self, embedding_dim, embedding_normalization):
        super().__init__()
        if embedding_normalization not in ("none", "l2"):
            raise ValueError(
                "embedding_normalization must be either 'none' or 'l2'")
        self.embedding_normalization = embedding_normalization
        self.backbone = ResNet50Conv4cBackbone()
        self.temporal_embedder = TCCTemporalEmbedder(embedding_dim)

    def forward(self, frames):
        if frames.ndim != 6:
            raise ValueError(
                "TCC frames must have shape [B, 1, 2, 3, 224, 224]")
        batch, clip_len, context_size, channels, height, width = frames.shape
        if (clip_len, context_size, channels, height, width) != (
                1, 2, 3, 224, 224):
            raise ValueError(
                "TCC frames must have shape [B, 1, 2, 3, 224, 224], "
                f"got {tuple(frames.shape)}")

        flat = frames.reshape(batch * clip_len * context_size,
                              channels, height, width)
        features = self.backbone(flat)
        features = features.reshape(batch, clip_len, context_size,
                                    *features.shape[1:])
        embeddings = self.temporal_embedder(features)
        if self.embedding_normalization == "l2":
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1,
                                     eps=1.0e-12)
        return embeddings
