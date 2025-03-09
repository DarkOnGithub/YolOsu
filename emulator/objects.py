import utils.curves as curve

class HitObject:
    def __init__(self, x, y, time):
        self.x = x 
        self.y = y
        self.time = time


class HitCircle(HitObject):
    def __init__(self, x, y, time):
        super().__init__(x, y, time)
        
class Slider(HitObject):
    def __init__(self, x, y, time, curve_type, curve_points, length):
        super().__init__(x, y, time)
        curve_type = curve_type.lower()
        if curve_type == "b":
            self.curve = curve.Bezier(curve_points)
        elif curve_type == "c":
            self.curve = curve.Catmull(curve_points)
        