import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = CNNBlock(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = CNNBlock(mid_channels, mid_channels, 3, 1, 1)
        self.shortcut = CNNBlock(in_channels, mid_channels, 1, 1, 0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        s = self.shortcut(x)
        return torch.cat([s, y], dim=1)
    
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = CNNBlock(3, 32, 3, 2, 1)  # Stride 2: 320x320
        self.stage1 = self.make_stage(32, 64, 1)  # Stride 2
        self.conv2 = CNNBlock(64, 128, 3, 2, 1)  # Stride 4: 160x160
        self.stage2 = self.make_stage(128, 128, 2)  # Stride 4
        self.conv3 = CNNBlock(128, 256, 3, 2, 1)  # Stride 8: 80x80
        self.stage3 = self.make_stage(256, 256, 2)  # Stride 8
        self.conv4 = CNNBlock(256, 512, 3, 2, 1)  # Stride 16: 40x40
        self.stage4 = self.make_stage(512, 512, 1)  # Stride 16
        self.conv5 = CNNBlock(512, 1024, 3, 2, 1)  # Stride 32: 20x20
        self.stage5 = self.make_stage(1024, 1024, 1)  # Stride 32

    def make_stage(self, in_channels, out_channels, num_blocks):
        layers = [CSPBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(CSPBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)          # Stride 2
        x = self.stage1(x)         # Stride 2
        x = self.conv2(x)          # Stride 4
        c2 = self.stage2(x)        # Stride 4, 128 channels
        x = self.conv3(c2)         # Stride 8
        c3 = self.stage3(x)        # Stride 8, 256 channels
        x = self.conv4(c3)         # Stride 16
        c4 = self.stage4(x)        # Stride 16, 512 channels
        x = self.conv5(c4)         # Stride 32
        c5 = self.stage5(x)        # Stride 32, 1024 channels
        return c2, c3, c4, c5
    
    
class FPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024], out_channels=256):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            CNNBlock(out_channels, out_channels, 3, 1, 1) for _ in range(len(in_channels_list))
        ])

    def forward(self, inputs):
        # Lateral connections
        laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        # Top-down path
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += nn.functional.interpolate(laterals[i + 1], scale_factor=2, mode='nearest')
        # Post-processing
        outputs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        return outputs
    
    
class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3, num_classes=80, num_mask_coeffs=32):
        super(DetectionHead, self).__init__()
        self.conv1 = CNNBlock(in_channels, in_channels, 3, 1, 1)
        self.conv2 = CNNBlock(in_channels, in_channels, 3, 1, 1)
        self.head = nn.Conv2d(
            in_channels,
            num_anchors * (4 + 1 + num_classes + num_mask_coeffs),
            1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.head(x)
    
class PrototypeHead(nn.Module):
    def __init__(self, in_channels=128, num_prototypes=32):
        super(PrototypeHead, self).__init__()
        self.conv1 = CNNBlock(in_channels, 256, 3, 1, 1)
        self.conv2 = CNNBlock(256, 256, 3, 1, 1)
        self.head = nn.Conv2d(256, num_prototypes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.head(x)
    
class YOLOv8Segmentation(nn.Module):
    def __init__(self, num_classes=80, num_anchors=3, num_mask_coeffs=32, num_prototypes=32):
        super(YOLOv8Segmentation, self).__init__()
        self.backbone = Backbone()
        self.fpn = FPN(in_channels_list=[256, 512, 1024], out_channels=256)
        self.detection_heads = nn.ModuleList([
            DetectionHead(256, num_anchors, num_classes, num_mask_coeffs) for _ in range(3)
        ])
        self.prototype_head = PrototypeHead(128, num_prototypes)

    def forward(self, x):
        # Extract features from backbone
        c2, c3, c4, c5 = self.backbone(x)
        # Enhance features with FPN
        p3, p4, p5 = self.fpn([c3, c4, c5])
        # Detection predictions
        det_outputs = [head(p) for head, p in zip(self.detection_heads, [p3, p4, p5])]
        # Mask prototypes
        prototypes = self.prototype_head(c2)
        return det_outputs, prototypes