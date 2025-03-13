import cv2
import time

input_video_path = input("Video path: ")
output_video_path = "output_video_small.mp4"  


cap = cv2.VideoCapture(input_video_path)


if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video FPS: {fps}")
print(f"Resolution: {frame_width}x{frame_height}")
print(f"Total Frames: {frame_count}")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


frame_number = 0

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    
    current_time = frame_number / fps
    
    
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)
    milliseconds = int((current_time % 1) * 1000)
    
    
    timestamp = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d} | F:{frame_number}"
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  
    thickness = 1     
    
    
    text_size = cv2.getTextSize(timestamp, font, font_scale, thickness)[0]
    
    
    text_x = frame_width - text_size[0] - 5
    text_y = 15
    
    
    cv2.rectangle(
        frame,
        (text_x - 2, text_y - text_size[1] - 2),
        (text_x + text_size[0] + 2, text_y + 2),
        (0, 0, 0),  
        -1          
    )
    
    cv2.putText(
        frame,
        timestamp,
        (text_x, text_y),
        font,
        font_scale,
        (0, 255, 0),  
        thickness,
        cv2.LINE_AA
    )
    
    
    out.write(frame)
    
    
    
    frame_number += 1
    
    
    delay = int(1000 / fps)


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing completed. Output saved as '{output_video_path}'")
print(f"Processed {frame_number} frames")