import zipfile
import tomllib
from emulator.objects import HitCircle, Slider
import utils.utils as utils
def extract_osu_file(osz_path):
    with zipfile.ZipFile(osz_path, "r") as z:
            file_list = z.namelist()
            
            target_files = [f for f in file_list if f.lower().endswith(".osu")]
            
            for file in target_files:
                with z.open(file) as f:
                    content = f.read().decode('utf-8', errors='replace')
                    return content
    return None

def parse_osu_file(osu_content):
    if not osu_content:
        return None
        
    sections = {}
    current_section = None
    
    for line in osu_content.splitlines():
        line = line.strip()
        
        if not line:
            continue
            
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            sections[current_section] = []
            continue
            
        if current_section:
            sections[current_section].append(line)
    
    return sections    

def parse_hit_objects(hit_objects):
    objects = []
    for line in hit_objects:
        x, y, time, type, _, *params = line.split(",")
        x, y = int(x), int(y)
        type = int(type)
        slider = utils.nth_bit_set(type, 1)
        hit_circle = utils.nth_bit_set(type, 0)
    
        if slider and hit_circle:
            print("??")
            continue
        
        if hit_circle:
            objects.append(HitCircle(x, y, time))

        if slider:
            curve_info = params[0].split("|")
            curve_type = curve_info[0]
            curve_points = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in curve_info[1:]]
            length = int(params[1])
            objects.append(Slider(x, y, time, curve_type, curve_points, length))
    return objects

def parse_osz_file(osz_path):
    content = extract_osu_file(osz_path)
    if content is None:
        return None
    return parse_hit_objects(parse_osu_file(content)['HitObjects'])    
        