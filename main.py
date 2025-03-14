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
    if not hit_objects:
        print("No hit objects to process.")
        return
    
    # Calculate offset based on first hit object's appearance time
    first_obj = hit_objects[0]
    first_appear = first_obj.time - approach_time_ms
    pre_roll = 1000  # 1 second before first approach circle appears
    desired_start_time = max(0, first_appear - pre_roll)
    offset_ms = -desired_start_time
    print(f"Adjusted offset to {offset_ms}ms. First object appears at {first_appear}ms, video starts at {desired_start_time}ms")
    
    ms_per_frame = 1000 / fps
    frame_index = 0
    temporal_buffer_ms = max(ms_per_frame * 1.5, 5)  
    
    # Seek to the desired start time if possible
    desired_start_frame = int(desired_start_time * fps / 1000)
    cap.set(cv2.CAP_PROP_POS_FRAMES, desired_start_frame)
    frame_index = desired_start_frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = frame_index * ms_per_frame - offset_ms
        frame_end_time = frame_start_time + ms_per_frame
        
        mask = np.zeros((height, width), dtype=np.uint8)
        approach_mask = np.zeros((height, width), dtype=np.uint8)
        
        visible_objects = []
        for obj in hit_objects:
            appear_time = obj.time - approach_time_ms
            
            if isinstance(obj, Slider):
                hit_end_time = obj.time + obj.ball.total_duration
            else:
                hit_end_time = obj.time
            
            if (appear_time - temporal_buffer_ms <= frame_end_time and 
                hit_end_time + temporal_buffer_ms >= frame_start_time):
                visible_objects.append(obj)
            
            if appear_time > frame_end_time + approach_time_ms:
                break
        
        for obj in visible_objects:
            appear_time = obj.time - approach_time_ms
            
            base_mask = obj.get_segmentation_mask(width, height, object_radius)
            mask = np.maximum(mask, base_mask)
            
            if isinstance(obj, Slider):
                ball_mask = obj.ball.get_segmentation_mask(width, height, frame_start_time, int(object_radius) * 2)
                if np.any(ball_mask):
                    mask = np.maximum(mask, ball_mask)
            
            approach_circle = ApproachCircle((obj.x, obj.y), obj.time, 
                                            approach_time_ms + temporal_buffer_ms, 
                                            object_radius)
            thickness = 2
            circle_mask = approach_circle.get_segmentation_mask(width, height, frame_start_time, thickness)
            approach_mask = np.maximum(approach_mask, circle_mask)
        
        if np.any(mask) or np.any(approach_mask):
            white_mask = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            combined_white = np.maximum(mask, approach_mask)
            resized_white = cv2.resize(combined_white, (video_width, video_height))
            white_mask[resized_white > 0] = (255, 255, 255)
            
            cv2.putText(frame, f"Frame: {frame_index} Time: {frame_start_time:.0f}ms", 
                      (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            alpha = 0.5
            frame = cv2.addWeighted(frame, 1, white_mask, alpha, 0)
        
        out.write(frame)
        frame_index += 1
        
        if frame_index % 100 == 0:
            print(f"  Processed {frame_index}/{frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Video processing complete: {output_path}")
    
if __name__ == "__main__":
    overlay_objects_on_video(
        "2285243 Jeff Williams feat. Casey Lee Williams - Time to Say Goodbye (TV Size)",
        "No Return", 
        "output_with_objects.mp4"
    )

