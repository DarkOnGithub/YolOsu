import utils.curves as curve
import utils.utils as utils
class HitObject:
    def __init__(self, x, y, time):
        self.x, self.y = utils.osu_to_screen(x, y)
        self.time = int(time)


class HitCircle(HitObject):
    def __init__(self, x, y, time):
        super().__init__(x, y, time)
        print(self.x, self.y, self.time)  
class Slider(HitObject):
    def __init__(self, x, y, time, curve_type, curve_points, length):
        super().__init__(x, y, time)
        curve_type = curve_type.lower()
        if curve_type == "b":
            self.curve = curve.Bezier(curve_points)
        elif curve_type == "c":
            self.curve = curve.Catmull(curve_points)
        self.curve_points = curve_points
        self.length = length
        