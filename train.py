import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, 
                                os.path.splitext(self.image_files[idx])[0] + '.txt')
        print(label_path)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        
        masks = []
        boxes = []
        classes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                polygon = np.array(parts[1:]).reshape(-1, 2)
                
                
                polygon[:, 0] *= orig_w
                polygon[:, 1] *= orig_h
                polygon = polygon.astype(np.int32)
                
                
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 1)
                
                
                x_min, y_min = np.min(polygon, axis=0)
                x_max, y_max = np.max(polygon, axis=0)
                
                
                x_center = (x_min + x_max) / 2 / orig_w
                y_center = (y_min + y_max) / 2 / orig_h
                width = (x_max - x_min) / orig_w
                height = (y_max - y_min) / orig_h
                
                masks.append(mask)
                boxes.append([x_center, y_center, width, height])
                classes.append(class_id)
        
        
        if self.transform:
            transformed = self.transform(image=image, masks=masks)
            image = transformed['image']
            masks = transformed['masks']
            
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)
        masks = torch.stack([torch.from_numpy(m) for m in masks]).float()
        
        return image, {'boxes': boxes, 'labels': classes, 'masks': masks}
    
import albumentations as A

train_transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append({
            'boxes': target['boxes'],
            'labels': target['labels'],
            'masks': target['masks']
        })
    return torch.stack(images), targets

train_dataset = SegmentationDataset(
    image_dir='osu_dataset/images',
    label_dir='osu_dataset/labels',
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.net import YOLOv8Seg, SegmentationLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv8Seg(num_classes=80).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
loss_fn = SegmentationLoss(num_classes=80).to(device)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        
        
        processed_targets = []
        for t in targets:
            img_targets = []
            for box, label, mask in zip(t['boxes'], t['labels'], t['masks']):
                
                mask = mask.cpu().numpy()
                img_targets.append([
                    0,  
                    label.item(),
                    box[0].item(),  
                    box[1].item(),  
                    box[2].item(),  
                    box[3].item(),  
                    mask
                ])
            processed_targets.append(img_targets)
        
        
        optimizer.zero_grad()
        outputs = model(images)
        
        
        loss, loss_components = loss_fn(outputs, processed_targets)
        
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}')
    
    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1} completed | Avg Loss: {epoch_loss:.4f}')