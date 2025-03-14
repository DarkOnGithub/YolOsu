import os
import cv2
import numpy as np
from parser.osz_parser import parse_osz_file
import utils.curves as curve
import sys
from emulator.objects import HitCircle, Slider

sys.modules['curve'] = curve

def calculate_approach_time(ar):
    # Existing implementation
    if ar == 5:
        return 1200
    elif ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    else:
        return 1200 - 750 * (ar - 5) / 5

def create_dataset(osz_path, difficulty, output_dir="dataset"):
    
    # Create directories
    images_dir = os.path.join(output_dir, "images", "train")
    labels_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    width, height = 192, 144

    print(f"Parsing beatmap: {osz_path}")
    beatmap = parse_osz_file(osz_path)
    if not beatmap or not beatmap.difficulties:
        print("No difficulties found in beatmap")
        return
    video_path = beatmap.generate_clip(difficulty)
    if video_path is None:
        print("Difficulty not found in beatmap")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get hit objects
    difficulty = beatmap.difficulties[difficulty]
    hit_objects = sorted(difficulty.hit_objects, key=lambda obj: obj.time)
    cs = difficulty.difficulty.get("cs", 4)
    object_radius = (54.4 - 4.48 * cs) * (0.8 * video_height) / 384
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Save frame
        cv2.imwrite(os.path.join(images_dir, f"{frame_index:06}.jpg"), frame)
        
        # Create label file
        label_path = os.path.join(labels_dir, f"{frame_index:06}.txt")
        with open(label_path, 'w') as f:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            for obj in hit_objects:
                if isinstance(obj, (HitCircle, Slider)):
                    appear_time = obj.time - calculate_approach_time(difficulty.difficulty.get("ar", 5))
                    if not (appear_time <= current_time <= obj.time + 1000):
                        continue
                    
                    # Generate mask
                    mask = obj.get_segmentation_mask(video_width, video_height, object_radius)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process contours
                    for contour in contours:
                        # Simplify contour
                        epsilon = 0.002 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
                        
                        # Normalize coordinates
                        normalized = approx / [video_width, video_height]
                        
                        # Format: class_id x1 y1 x2 y2 ...
                        class_id = 0 if isinstance(obj, HitCircle) else 1
                        points_str = " ".join([f"{x:.6f} {y:.6f}" for (x, y) in normalized])
                        f.write(f"{class_id} {points_str}\n")

        frame_index += 1

    cap.release()
    print(f"Dataset created with {frame_index} frames")

import os
import cv2
import numpy as np

def create_verification_video(dataset_path, output_video="dataset_verify.mp4", fps=30):
    # Path configuration
    images_dir = os.path.join(dataset_path, "images", "train")
    labels_dir = os.path.join(dataset_path, "labels", "train")
    
    # Get sorted list of images
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    if not image_files:
        print("No images found in dataset directory!")
        return

    # Get video properties from first image
    sample_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width = sample_image.shape[:2]
    
    # Create video writer (triple width for comparison)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width*3, height))

    # Color palette for classes (BGR format)
    class_colors = {
        0: (0, 255, 0),   # HitCircle - Green
        1: (255, 0, 0),   # Slider - Blue
        2: (0, 0, 255)     # ApproachCircle - Red
    }

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        frame = cv2.imread(img_path)
        
        label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    class_id = int(parts[0])
                    points = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                    
                    points[:, 0] *= width
                    points[:, 1] *= height
                    points = points.astype(np.int32)
                    color = class_colors.get(class_id, (255, 255, 255))
                    cv2.fillPoly(mask, [points], color)
        
        overlay = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        
        combined = np.hstack([
            frame,          # Original image
            mask,           # Pure mask
            overlay         # Overlay
        ])
        
        # Add frame info
        text = f"Frame: {idx} | File: {img_file}"
        cv2.putText(combined, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        video_writer.write(combined)
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_files)} frames...")

    video_writer.release()
    print(f"Verification video created: {output_video}")

if __name__ == "__main__":
    create_verification_video(
        dataset_path="dataset",
        output_video="dataset_verification.mp4",
        fps=60  # Match original game FPS
    )