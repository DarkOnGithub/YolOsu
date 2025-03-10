import os
import cv2
import numpy as np
from parser.osz_parser import parse_osz_file
import utils.curves as curve  
import sys

sys.modules['curve'] = curve

def calculate_approach_time(ar):

    if ar < 5:
        return 1800 - 120 * ar
    else:
        return 1200 - 150 * (ar - 5)

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

def overlay_objects_on_video(osz_path, difficulty, video_path, output_path="output_video.mp4"):

    width, height = 192, 144
    
    
    print(f"Parsing beatmap: {osz_path}")
    beatmap = parse_osz_file(osz_path)
    if not beatmap or not beatmap.difficulties:
        print("No difficulties found in beatmap")
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
    
    offset_ms = -hit_objects[0].time + 1700

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time_ms = frame_index * ms_per_frame - offset_ms
        
        frame_mask = np.zeros((height, width), dtype=np.uint8)
        
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
                progress = (current_time_ms - appear_time) / approach_time_ms

                obj_mask = np.zeros((height, width), dtype=np.uint8)
                
                base_mask = obj.get_segmentation_mask(width, height, object_radius)
                
                obj_mask = np.maximum(obj_mask, base_mask)
                
                approach_scale = 3.0 - 2.0 * progress
                approach_radius = object_radius * approach_scale
                
                if approach_scale > 1.05:  
                    approach_mask = np.zeros((height, width), dtype=np.uint8)
                    
                    if isinstance(obj, HitCircle):
                        x, y = round(obj.x), round(obj.y)
                        cv2.circle(approach_mask, (x, y), int(approach_radius), 1, 2)
                    elif isinstance(obj, Slider):
                        x, y = round(obj.curve_points[0][0]), round(obj.curve_points[0][1])
                        cv2.circle(approach_mask, (x, y), int(approach_radius), 1, 2)
                    
                    obj_mask = np.maximum(obj_mask, approach_mask)
                
                frame_mask = np.maximum(frame_mask, obj_mask)

            obj_index += 1

        if np.any(frame_mask):
            frame_mask = cv2.resize(frame_mask, (video_width, video_height))
            
            colored_mask = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            colored_mask[frame_mask > 0] = [0, 0, 255] 
            
            alpha = 0.5
            frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
        
        out.write(frame)
        frame_index += 1
        
        if frame_index % 100 == 0:
            print(f"  Processed {frame_index}/{frame_count} frames...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    out.release()
    print(f"Video processing complete: {output_path}")

if __name__ == "__main__":
    from emulator.objects import HitCircle, Slider
    

    overlay_objects_on_video(
        "maps/320118 Reol - No title.osz",
        "jieusieu's Lemur", 
        "danser_2025-03-10_22-12-28.mp4", 
        "output_with_objects.mp4"
    )