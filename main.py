import os
import re
import cv2
import numpy as np
from parser.osz_parser import parse_osz_file
import utils.curves as curve  
import sys
from emulator.objects import HitCircle, Slider,ApproachCircle 
import utils.utils as utils
sys.modules['curve'] = curve

def calculate_approach_time(ar):
    if ar == 5:
        return 1200
    elif ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    else:
        return 1200 - 750 * (ar - 5) / 5

def generate_masks_from_beatmap(osz_path, output_dir="beatmap_masks", radius=10):

    os.makedirs(output_dir, exist_ok=True)
    
    width, height = 192, 144
    
    print(f"Parsing beatmap: {osz_path}")
    beatmap = parse_osz_file(osz_path)
    print(beatmap)
    if not beatmap or not beatmap.difficulties:
        print("No difficulties found in beatmap")
        return
    
    for diff_name, difficulty in beatmap.difficulties.items():
        print(f"Processing difficulty: {diff_name}")
        
        diff_dir = os.path.join(output_dir, diff_name)
        os.makedirs(diff_dir, exist_ok=True)
        
        cs = difficulty.difficulty.get("cs", 4)
        object_radius = 54.4 - 4.48 * cs
        scale_factor = (0.8 * height) / 384
        object_radius = object_radius * scale_factor
        print(f"  Circle size: {cs}, radius: {object_radius}")
        for i, obj in enumerate(difficulty.hit_objects):
            mask = obj.get_segmentation_mask(width, height, object_radius)
            
            object_type = "circle" if isinstance(obj, HitCircle) else "slider"
            mask_filename = os.path.join(diff_dir, f"{i}.png")
            cv2.imwrite(mask_filename, mask * 255)
            
        
            if i % 20 == 0:
                print(f"  Processed {i} objects...")
        
        print(f"  Completed {len(difficulty.hit_objects)} objects for difficulty {diff_name}")
    
    print(f"All masks saved to {output_dir} directory")

def overlay_objects_on_video(osz_path, difficulty, output_path="output_video.mp4"):
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    diff_name, difficulty = difficulty, beatmap.difficulties.get(difficulty, None)
    if not difficulty:
        print(f"Error: Difficulty {difficulty} not found in beatmap")
        return
    
    print(f"Processing difficulty: {diff_name} for video overlay")
    
    ar = difficulty.difficulty.get("ar", 5)
    approach_time_ms = calculate_approach_time(ar)
    print(f"  Approach rate: {ar}, visible for {approach_time_ms}ms")
    
    cs = difficulty.difficulty.get("cs", 4)
    object_radius = 54.4 - 4.48 * cs
    scale_factor = (0.8 * height) / 384
    object_radius = object_radius * scale_factor
    
    hit_objects = sorted(difficulty.hit_objects, key=lambda obj: obj.time)
    ms_per_frame = 1000 / fps
    frame_index = 0
    current_obj_index = 0
    offset_ms = -hit_objects[0].time + 1000 + min(1800, approach_time_ms)
    
    
    hit_time_window = 50  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time_ms = frame_index * ms_per_frame - offset_ms
        
        
        mask = np.zeros((height, width), dtype=np.uint8)
        approach_mask = np.zeros((height, width), dtype=np.uint8)
        hit_time_mask = np.zeros((height, width), dtype=np.uint8)  
        
        while current_obj_index < len(hit_objects) and hit_objects[current_obj_index].time < current_time_ms:
            current_obj_index += 1
        
        obj_index = current_obj_index
        while obj_index < len(hit_objects):
            obj = hit_objects[obj_index]
            appear_time = obj.time - approach_time_ms
            hit_time = obj.time
            
            if appear_time > current_time_ms:
                break
                
            if current_time_ms <= hit_time:
                
                base_mask = obj.get_segmentation_mask(width, height, object_radius)
                
                
                if abs(current_time_ms - hit_time) <= hit_time_window:
                    hit_time_mask = np.maximum(hit_time_mask, base_mask)
                else:
                    mask = np.maximum(mask, base_mask)
                
                
                approach_circle = ApproachCircle((obj.x, obj.y), obj.time, approach_time_ms, object_radius)
                circle_mask = approach_circle.get_segmentation_mask(width, height, current_time_ms)
                
                
                if abs(current_time_ms - hit_time) <= hit_time_window:
                    hit_time_mask = np.maximum(hit_time_mask, circle_mask)
                else:
                    approach_mask = np.maximum(approach_mask, circle_mask)
            
            obj_index += 1
        
        if np.any(mask) or np.any(approach_mask) or np.any(hit_time_mask):
            white_mask = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            blue_mask = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            
            if np.any(mask) or np.any(approach_mask):
                combined_white = np.maximum(mask, approach_mask)
                resized_white = cv2.resize(combined_white, (video_width, video_height))
                white_mask[resized_white > 0] = (255, 255, 255)  
            
            if np.any(hit_time_mask):
                resized_blue = cv2.resize(hit_time_mask, (video_width, video_height))
                blue_mask[resized_blue > 0] = (255, 0, 0)  
            
            cv2.putText(frame, f"Frame: {frame_index} Time: {current_time_ms:.0f}ms", 
                      (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            
            alpha = 0.5
            frame = cv2.addWeighted(frame, 1, white_mask, alpha, 0)
            frame = cv2.addWeighted(frame, 1, blue_mask, alpha, 0)
        
        out.write(frame)
        frame_index += 1
        
        if frame_index % 100 == 0:
            print(f"  Processed {frame_index}/{frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Video processing complete: {output_path}")
    
if __name__ == "__main__":
    overlay_objects_on_video(
        "320118 Reol - No title",
        "jieusieu's Lemur", 
        "output_with_objects.mp4"
    )

