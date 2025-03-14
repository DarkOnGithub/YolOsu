























    






























































            






















                







































                









































































import os
import re
import cv2
import numpy as np
from parser.osz_parser import parse_osz_file
import utils.curves as curve
import sys
from emulator.objects import HitCircle, Slider, ApproachCircle
import utils.utils as utils

sys.modules['curve'] = curve
CONTOUR_APPROXIMATION_EPSILON = 0.0001
ITEMS_ID = {
    'hit_circle': 0,
    "slider": 1,
    'approach_circle': 2,
    'slider_ball': 3,
}


def calculate_approach_time(ar):
    if ar == 5:
        return 1200
    elif ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    else:
        return 1200 - 750 * (ar - 5) / 5
def export_polygon(contour):
    epsilon = CONTOUR_APPROXIMATION_EPSILON * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.squeeze().reshape(-1, 2).tolist()

def create_dataset(osz_path, difficulty, output_dir="dataset", resolution=(192, 144)):
    width, height = 192, 144
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

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

    
    first_obj = hit_objects[0]
    first_appear = first_obj.time - approach_time_ms
    pre_roll = 1000  
    desired_start_time = max(0, first_appear - pre_roll)
    offset_ms = -desired_start_time
    print(
        f"Adjusted offset to {offset_ms}ms. First object appears at {first_appear}ms, video starts at {desired_start_time}ms")

    ms_per_frame = 1000 / fps
    frame_index = 0
    temporal_buffer_ms = max(ms_per_frame * 1.5, 5)

    
    desired_start_frame = int(desired_start_time * fps / 1000)
    cap.set(cv2.CAP_PROP_POS_FRAMES, desired_start_frame)
    frame_index = desired_start_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        instances = []
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
            contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                instances.append((ITEMS_ID['hit_circle'], export_polygon(max_contour)))
                
            if isinstance(obj, Slider):
                ball_mask = obj.ball.get_segmentation_mask(width, height, frame_start_time, int(object_radius))
                if np.any(ball_mask):
                    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        instances.append((ITEMS_ID['slider_ball'], export_polygon(max_contour)))

            approach_circle = ApproachCircle((obj.x, obj.y), obj.time,
                                             approach_time_ms + temporal_buffer_ms,
                                             object_radius)
            thickness = 2
            circle_mask = approach_circle.get_segmentation_mask(width, height, frame_start_time, thickness)
            contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                instances.append((ITEMS_ID['approach_circle'], export_polygon(max_contour)))
            
        if instances:
            img_path = os.path.join(images_dir, f"{frame_index:06}.jpg")
            cv2.imwrite(img_path, frame)

            label_path = os.path.join(labels_dir, f"{frame_index:06}.txt")
            with open(label_path, 'w') as f:
                for class_id, points in instances:
                    normalized = []
                    for x, y in points:
                        nx = x / resolution[0]
                        ny = y / resolution[1]
                        normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])
                    f.write(f"{class_id} {' '.join(normalized)}\n")
                    

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"  Processed {frame_index}/{frame_count} frames...")

    cap.release()
def render_video_from_dataset(dataset_dir, output_path="rendered.mp4", fps=60):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    
    frames = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    if not frames:
        raise ValueError("No frames found in dataset directory")

    
    first_frame = cv2.imread(os.path.join(images_dir, frames[0]))
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    
    colors = {
        0: (0, 255, 0),   
        1: (255, 0, 0),   
        2: (0, 255, 255),  
        3: (255, 255, 0)  
    }
    line_thickness = 2  

    for frame_file in frames:
        frame = cv2.imread(os.path.join(images_dir, frame_file))
        label_path = os.path.join(labels_dir, frame_file.replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    points = list(map(float, parts[1:]))

                    
                    polygon = []
                    for i in range(0, len(points), 2):
                        x = int(points[i] * w)
                        y = int(points[i+1] * h)
                        polygon.append([x, y])

                    
                    if len(polygon) > 2:
                        np_poly = np.array([polygon], dtype=np.int32)
                        cv2.polylines(frame, [np_poly], True, colors[class_id], line_thickness,
                                    lineType=cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"Contour video rendered to {output_path}")

if __name__ == "__main__":
    create_dataset(
        "2285243 Jeff Williams feat. Casey Lee Williams - Time to Say Goodbye (TV Size)",
        "No Return",
        output_dir="osu_dataset"
    )

    render_video_from_dataset(
        "osu_dataset",
        output_path="rendered.mp4",
        fps=60
    )