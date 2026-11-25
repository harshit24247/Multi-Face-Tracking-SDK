"""Shared constants, model, loss and metric for the 68-point landmark model."""
import numpy as np, torch, torch.nn as nn, timm

S = 192  # crop size (agreed with Teams A and C)
REGIONS = {"jaw": (0, 17), "brows": (17, 27), "nose": (27, 36),
           "eyes": (36, 48), "lips": (48, 68)}  # 0-based, matches 17/10/9/12/20 split


def _flip_index():
    pairs = [(i, 16 - i) for i in range(8)]
    pairs += [(17, 26), (18, 25), (19, 24), (20, 23), (21, 22), (31, 35), (32, 34)]
    pairs += [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)]
    pairs += [(48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]
    idx = list(range(68))
    for a, b in pairs:
        idx[a], idx[b] = b, a
    return np.array(idx)


FLIP = _flip_index()


class LandmarkNet(nn.Module):
    """MobileNetV2-0.5 backbone (ReLU6, NPU friendly) -> 136 normalized coords."""
    def __init__(self, backbone="mobilenetv2_050", pretrained=True):
        super().__init__()
        self.b = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = nn.Linear(self.b.num_features, 136)

    def forward(self, x):
        return self.head(self.b(x))


def wing_loss(pred, target, w=0.05, eps=0.01):
    d = (pred - target).abs()
    c = w - w * np.log(1 + w / eps)
    return torch.where(d < w, w * torch.log(1 + d / eps), d - c).mean()


def nme_per_point(pred, target):
    """Per-point error normalised by inter-ocular distance (outer eye corners 36, 45)."""
    p, t = pred.reshape(-1, 68, 2), target.reshape(-1, 68, 2)
    iod = (t[:, 36] - t[:, 45]).norm(dim=1, keepdim=True)
    return (p - t).norm(dim=2) / iod  # (B, 68)