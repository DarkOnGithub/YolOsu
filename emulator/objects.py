from turtle import width
import numpy as np
import math
import cv2
import utils.curves as curve
from utils.utils import osu_pixels_to_normal_coords

class HitObject:
    def __init__(self, x, y, time, resolution_width=192, resolution_height=144):
        
        self.osu_x, self.osu_y = x, y
        self.time = int(time)
        
        
        self.x, self.y = osu_pixels_to_normal_coords(x, y, resolution_width, resolution_height)
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
    
    def convert_point(self, x, y):
        return osu_pixels_to_normal_coords(x, y, self.resolution_width, self.resolution_height)
    
    def get_segmentation_polygon(self, r):
        raise NotImplementedError
        
    def get_segmentation_mask(self, width, height, r):
        raise NotImplementedError


class HitCircle(HitObject):
    def __init__(self, x, y, time, resolution_width=192, resolution_height=144):
        super().__init__(x, y, time, resolution_width, resolution_height)
    def get_segmentation_mask(self, width, height, r):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        center_x, center_y = int(round(self.x)), int(round(self.y))
        radius = int(r)
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)
        return mask

class Slider(HitObject):
    def __init__(self, x, y, time, curve_type, curve_points, length, beat_duration, repeats=1, velocity_multiplier=1.0,
                resolution_width=192, resolution_height=144):
        super().__init__(x, y, time, resolution_width, resolution_height)
        self.curve_type = curve_type.lower()
        self.curve_points = [self.convert_point(x, y) for x, y in curve_points]
        self.slider_multiplier = 1.4  
        self.velocity_multiplier = velocity_multiplier
        self.beat_duration = beat_duration
        
        self.original_length = float(length)
        self.desired_length = self.calculate_scaled_length(length)
        
        scale_factor = (self.desired_length / self.original_length) if self.original_length != 0 else 1.0
        scaled_curve_points = [
            (
                self.x + (point[0] - self.x) * scale_factor,
                self.y + (point[1] - self.y) * scale_factor
            )
            for point in self.curve_points
        ]
        full_points = [(self.x, self.y)] + scaled_curve_points
        
        self.curve = self.create_curve(full_points)
        self.actual_length = self.curve.get_length()
        self.repeats = repeats
        self.ball = SliderBall(self, velocity_multiplier)
        
    def calculate_scaled_length(self, length):
        scale_factor = (0.8 * self.resolution_height) / 384
        return float(length) * scale_factor

    def create_curve(self, points):
        curve_map = {
            'b': curve.Bezier,
            'c': curve.Catmull,
            'l': curve.Linear,
            'p': curve.Perfect
        }
        curve_class = curve_map.get(self.curve_type, curve.Bezier)
        return curve_class(points)

    def get_segmentation_mask(self, width, height, radius):
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.curve.get_length() <= 0:
            return mask
        
        num_samples = self.calculate_sample_count(radius)
        points = []
        
        for t in np.linspace(0, self.repeats, num_samples):
            segment_progress = t % 1.0
            repeat_number = int(t)
            
            if repeat_number % 2 == 1:
                segment_progress = 1.0 - segment_progress
                
            x, y = self.ball.get_position_at_progress(segment_progress)
            x = int((x))
            y = int((y))
            
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
        
        radius_int = int(round(radius))
        for x, y in points:
            print(x,y)
            cv2.circle(mask, (x, y), radius_int, 1, -1)
        
        return mask

    def calculate_sample_count(self, radius):
        base_samples = int(self.desired_length / max(radius, 1)) * 2
        return np.clip(base_samples * self.repeats, 10, 1000)

class ApproachCircle:
    def __init__(self, center, time, approach_time_ms, radius=10):
        self.center = center
        self.time = time
        self.radius = radius            
        self.appear = time - approach_time_ms
        
    def get_segmentation_mask(self, width, height, time, thickness=1):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        
        buffer_time = 20  
        if time < (self.appear - buffer_time) or time >= (self.time + buffer_time):
            return mask
            
        progress = (time - self.appear) / (self.time - self.appear)
        progress = max(0.0, min(1.0, progress))  
        
        scale = 4.0 - 3.0 * progress
        
        current_radius = int(round(self.radius * scale))
        
        center_x = int((self.center[0]))
        center_y = int((self.center[1]))
        
        
        if abs(self.time - time) < 20:
            cv2.circle(mask, (center_x, center_y), current_radius, 1, max(5, thickness))
        else:
            cv2.circle(mask, (center_x, center_y), current_radius, 1, thickness)
        
        return mask
    


class SliderBall:
    def __init__(self, slider, velocity_multiplier=1.0):
        self.slider = slider
        self.velocity_multiplier = velocity_multiplier
        
        # Calculate duration based on ACTUAL curve length
        slider_velocity = self.slider.slider_multiplier * 100 * self.velocity_multiplier
        self.duration_per_repeat = (self.slider.actual_length / slider_velocity) * self.slider.beat_duration
        self.total_duration = self.duration_per_repeat * self.slider.repeats
        
        self.start_time = self.slider.time
        self.end_time = self.start_time + self.total_duration
        
    def get_position_at_progress(self, progress):
        if progress <= 1.0:
            base_x, base_y = self.slider.curve.point_at(progress)
        else:
            # Extend beyond curve using end direction
            end_x, end_y = self.slider.curve.point_at(1.0)
            dx, dy = self.slider.curve.get_end_direction()
            extension_length = (progress - 1.0) * self.slider.actual_length
            base_x = end_x + dx * extension_length
            base_y = end_y + dy * extension_length
        
        # Clamp to screen boundaries
        final_x = max(0, min(base_x, self.slider.resolution_width - 1))
        final_y = max(0, min(base_y, self.slider.resolution_height - 1))
        return final_x, final_y
    def get_position_at_time(self, current_time):
        if current_time <= self.start_time:
            return self.slider.x, self.slider.y
            
        if current_time >= self.end_time:
            if self.slider.repeats % 2 == 0:
                return self.slider.x, self.slider.y
            else:
                return self.get_position_at_progress(1.0)
        
        elapsed_time = current_time - self.start_time
        normalized_time = elapsed_time / self.total_duration
        
        repeat_number = int(normalized_time * self.slider.repeats)
        repeat_progress = (normalized_time * self.slider.repeats) % 1.0
        
        if repeat_number % 2 == 1:
            repeat_progress = 1.0 - repeat_progress
            
        return self.get_position_at_progress(repeat_progress)
    
            
    def get_segmentation_mask(self, width, height, current_time, ball_radius):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if self.start_time - 5 <= current_time <= self.end_time + 5:
            x, y = self.get_position_at_time(current_time)
            x_int, y_int = int(round(x)), int(round(y))
            
            if 0 <= x_int < width and 0 <= y_int < height:
                cv2.circle(mask, (x_int, y_int), ball_radius, 1, -1)
                
        return mask