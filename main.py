# import torch
# from model.net import YoloV3, config

# def test():
#     num_classes = 20
#     model = YoloV3(classes=num_classes,config=config)
#     img_size = 416
#     x = torch.randn((2, 3, img_size, img_size))
#     out = model(x)
#     assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
#     assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
#     assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)

# test()
import cv2
import numpy as np
from parser.osz_parser import parse_osz_file
from emulator.objects import HitCircle, Slider

def create_video_with_bounding_boxes(objects, output_path="output.mp4", fps=60, 
                                     width=1080, height=1920, preview_ms=350):
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Sort objects by time
    objects = sorted(objects, key=lambda obj: obj.time)
    
    if not objects:
        print("No objects to render")
        return
        
    # Calculate the duration of the video
    last_obj_time = objects[-1].time
    total_frames = int((last_obj_time + 1000) * fps / 1000)  # Add 1 second at the end
    
    # Dictionary to track which objects need bounding boxes in each frame
    bbox_objects = {}
    hit_objects = {}
    
    # Pre-calculate which objects appear in which frames
    for obj in objects:
        # Convert time to frame number
        obj_frame = int(obj.time * fps / 1000)
        bbox_frame = int((obj.time - preview_ms) * fps / 1000)
        
        # Store which objects need bounding boxes in each frame
        if bbox_frame not in bbox_objects:
            bbox_objects[bbox_frame] = []
        bbox_objects[bbox_frame].append(obj)
        
        # Store which objects appear in each frame
        if obj_frame not in hit_objects:
            hit_objects[obj_frame] = []
        hit_objects[obj_frame].append(obj)
    
    # Render video frame by frame
    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Render bounding boxes for objects that need them in this frame
        if frame_num in bbox_objects:
            for obj in bbox_objects[frame_num]:
                # For HitCircle, render a square bounding box
                if isinstance(obj, HitCircle):
                    radius = 50  # Adjust based on your game settings
                    top_left = (int(obj.x) - radius, int(obj.y) - radius)
                    bottom_right = (int(obj.x) + radius, int(obj.y) + radius)
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    
                # For Slider, render a bounding box that covers the whole slider path
                elif isinstance(obj, Slider):
                    # Start with the initial position
                    min_x, min_y = int(obj.x), int(obj.y)
                    max_x, max_y = int(obj.x), int(obj.y)
                    
                    # Include all curve points to determine the bounding box
                    for point_x, point_y in obj.curve_points:
                        min_x = min(min_x, point_x)
                        min_y = min(min_y, point_y)
                        max_x = max(max_x, point_x)
                        max_y = max(max_y, point_y)
                    
                    # Add padding
                    padding = 30
                    cv2.rectangle(frame, (min_x - padding, min_y - padding), 
                                 (max_x + padding, max_y + padding), (0, 255, 0), 2)
        
        # Render actual hit objects that appear in this frame
        if frame_num in hit_objects:
            for obj in hit_objects[frame_num]:
                # Draw the actual hit object
                if isinstance(obj, HitCircle):
                    cv2.circle(frame, (int(obj.x), int(obj.y)), 40, (0, 0, 255), -1)
                elif isinstance(obj, Slider):
                    # Draw the slider start point
                    cv2.circle(frame, (int(obj.x), int(obj.y)), 40, (0, 0, 255), -1)
                    
                    # Draw the slider path
                    points = [(int(obj.x), int(obj.y))]
                    for point_x, point_y in obj.curve_points:
                        points.append((point_x, point_y))
                    
                    # Draw lines connecting all points
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i + 1], (255, 0, 0), 3)
        
        # Add frame to video
        video.write(frame)
        
        # Print progress
        if frame_num % 100 == 0:
            print(f"Rendering frame {frame_num}/{total_frames}")
    
    # Release video writer
    video.release()
    print(f"Video saved to {output_path}")

# Parse the osu file and create the video
obj = parse_osz_file("./maps/test.osz")
create_video_with_bounding_boxes(obj)