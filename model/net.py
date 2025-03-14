import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import math
class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        hidden_dim = out_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_dim, 1)
        self.conv2 = ConvBNSiLU(hidden_dim, out_channels, 3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        hidden_dim = out_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_dim, 1)
        self.conv2 = ConvBNSiLU(in_channels, hidden_dim, 1)
        self.conv3 = ConvBNSiLU(2 * hidden_dim, out_channels, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden_dim, hidden_dim, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_dim = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_dim, 1)
        self.conv2 = ConvBNSiLU(hidden_dim * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), dim=1))

class CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stem = ConvBNSiLU(3, 64, 6, 2, 2)  
        self.dark2 = nn.Sequential(
            ConvBNSiLU(64, 128, 3, 2, 1),        
            CSPBlock(128, 128, n=3)
        )
        self.dark3 = nn.Sequential(
            ConvBNSiLU(128, 256, 3, 2, 1),        
            CSPBlock(256, 256, n=6)
        )
        self.dark4 = nn.Sequential(
            ConvBNSiLU(256, 512, 3, 2, 1),        
            CSPBlock(512, 512, n=9)
        )
        self.dark5 = nn.Sequential(
            ConvBNSiLU(512, 1024, 3, 2, 1),       
            CSPBlock(1024, 1024, n=3),
            SPPF(1024, 1024, k=5)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x3 = self.dark3(x)  
        x4 = self.dark4(x3) 
        x5 = self.dark5(x4) 
        return [x3, x4, x5]

class PANet(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], depth=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.hidden_dim = in_channels[0]
        
        self.conv1 = ConvBNSiLU(in_channels[2], self.hidden_dim, 1)
        self.conv2 = ConvBNSiLU(self.hidden_dim * 2, self.hidden_dim, 3, padding=1)
        
        self.conv3 = ConvBNSiLU(self.hidden_dim, self.hidden_dim, 3, 2, 1)
        self.conv4 = ConvBNSiLU(self.hidden_dim * 2, self.hidden_dim, 3, padding=1)
        
        self.output_conv = ConvBNSiLU(self.hidden_dim * 3, self.hidden_dim, 3, padding=1)

    def forward(self, features):
        x3, x4, x5 = features
        
        
        p5 = self.conv1(x5)
        p4 = self.up(p5)
        p4 = torch.cat([p4, x4], 1)
        p4 = self.conv2(p4)
        
        
        p3 = self.conv3(p4)
        p3 = torch.cat([p3, x3], 1)
        p3 = self.conv4(p3)
        
        
        return self.output_conv(torch.cat([p5, p4, p3], 1))

class SegmentationHead(nn.Module):
    def __init__(self, num_classes, mask_dim=28):
        super().__init__()
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        
        
        self.box_conv = nn.Conv2d(256, 4, 1)
        self.obj_conv = nn.Conv2d(256, 1, 1)
        self.cls_conv = nn.Conv2d(256, num_classes, 1)
        
        
        self.mask_conv = nn.Conv2d(256, mask_dim**2, 1)
        
    def forward(self, x):
        box = self.box_conv(x)  
        obj = self.obj_conv(x)  
        cls = self.cls_conv(x)  
        mask = self.mask_conv(x)  
        
        
        mask = mask.view(mask.size(0), self.mask_dim, self.mask_dim, -1)
        mask = mask.permute(0, 3, 1, 2)  
        
        return box, obj, cls, mask

class YOLOv8Seg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck = PANet()
        self.head = SegmentationHead(num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features)
        return self.head(neck_features)

class CIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target):
        
        pred_xy = pred[..., :2]
        pred_wh = pred[..., 2:4]
        target_xy = target[..., :2]
        target_wh = target[..., 2:4]
        
        
        inter_xy_min = torch.max(pred_xy - pred_wh/2, target_xy - target_wh/2)
        inter_xy_max = torch.min(pred_xy + pred_wh/2, target_xy + target_wh/2)
        inter_wh = (inter_xy_max - inter_xy_min).clamp(0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        target_area = target_wh[..., 0] * target_wh[..., 1]
        union_area = pred_area + target_area - inter_area + self.eps
        
        
        iou = inter_area / union_area
        
        
        enclose_xy_min = torch.min(pred_xy - pred_wh/2, target_xy - target_wh/2)
        enclose_xy_max = torch.max(pred_xy + pred_wh/2, target_xy + target_wh/2)
        enclose_wh = (enclose_xy_max - enclose_xy_min).clamp(0)
        
        c2 = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2 + self.eps
        rho2 = (pred_xy[..., 0] - target_xy[..., 0]) ** 2 + \
               (pred_xy[..., 1] - target_xy[..., 1]) ** 2
        
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(pred_wh[..., 0] / (pred_wh[..., 1] + self.eps)) - 
            torch.atan(target_wh[..., 0] / (target_wh[..., 1] + self.eps)), 2)
        
        alpha = v / (v - iou + (1 + self.eps))
        
        return 1 - iou + (rho2 / c2) + v * alpha

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, mask_dim=28):
        super().__init__()
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        self.box_loss = CIoULoss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targets):
        box_pred, obj_pred, cls_pred, mask_pred = preds
        B, _, H, W = box_pred.shape
        device = box_pred.device
        
        total_box = 0.0
        total_obj = 0.0
        total_cls = 0.0
        total_mask = 0.0
        
        for batch_idx in range(B):
            
            batch_targets = [t for t in targets if t[0] == batch_idx]
            if len(batch_targets) == 0:
                continue
                
            
            gt_boxes = torch.stack([t[2:6] for t in batch_targets]).to(device)  
            gt_classes = torch.stack([t[1] for t in batch_targets]).to(device)  
            gt_masks = [t[6] for t in batch_targets]  
            
            
            grid_xy = (gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2  
            grid_xy = (grid_xy * torch.tensor([W, H], device=device)).long()
            
            for gt_idx, (xy, cls, box, mask) in enumerate(zip(grid_xy, gt_classes, gt_boxes, gt_masks)):
                x, y = xy
                if x >= W or y >= H or x < 0 or y < 0:
                    continue
                    
                
                pred_box = box_pred[batch_idx, :, y, x]
                total_box += self.box_loss(pred_box, box)
                
                
                total_obj += self.obj_loss(obj_pred[batch_idx, 0, y, x], torch.tensor(1.0, device=device))
                
                
                cls_target = F.one_hot(cls, self.num_classes).float()
                total_cls += self.cls_loss(cls_pred[batch_idx, :, y, x], cls_target)
                
                
                pred_mask = mask_pred[batch_idx, y * W + x]  
                x1, y1, x2, y2 = (box * torch.tensor([W, H, W, H], device=device)).int()
                cropped_mask = mask[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0).float()
                resized_mask = F.interpolate(cropped_mask, (self.mask_dim, self.mask_dim), 
                                           mode='bilinear', align_corners=False)
                resized_mask = (resized_mask > 0.5).float().squeeze()
                
                
                total_mask += self.mask_loss(pred_mask, resized_mask)
                
        
        total_loss = (total_box + total_obj + total_cls + total_mask) / B
        return total_loss, {
            'box': total_box / B,
            'obj': total_obj / B,
            'cls': total_cls / B,
            'mask': total_mask / B
        }