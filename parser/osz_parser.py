import zipfile
import tomllib
from emulator.objects import HitCircle, Slider
from emulator import Beatmap
import utils.utils as utils
import re


def extract_osu_file(osz_path):
    difficulties = {}
    with zipfile.ZipFile(osz_path, "r") as z:
            file_list = z.namelist()
            target_files = [f for f in file_list if f.lower().endswith(".osu")]
            for file in target_files:
                with z.open(file) as f:
                    difficulty = utils.get_difficulty(file)
                    if difficulty is None:
                        print(f"Could not find difficulty for {file}")
                        continue
                    content = f.read().decode('utf-8', errors='replace')
                    difficulties[difficulty] = content
    return difficulties

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
        x, y = utils.osu_pixels_to_normal_coords(int(x), int(y), 192, 144)
        type = int(type)
        slider = utils.nth_bit_set(type, 1)
        hit_circle = utils.nth_bit_set(type, 0)
        
        if hit_circle:
            objects.append(HitCircle(x, y, time))

        if slider:
            curve_info = params[0].split("|")
            curve_type = curve_info[0]
            curve_points = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in curve_info[1:]]
            length = int(params[1])
            objects.append(Slider(x, y, time, curve_type, curve_points, length))
            
    return objects

def parse_difficulty(sections):
    difficulty_section = sections.get("Difficulty")
    if difficulty_section:
        matches = re.findall(r":([\d.]+)", "\n".join(difficulty_section))
        return list(map(float, matches))
    return None



def parse_osz_file(osz_path):
    difficulties = extract_osu_file(osz_path)
    beatmap = Beatmap.Beatmap()
    for diff, content in difficulties.items():
        sections = parse_osu_file(content)
        hit_objects = sections.get("HitObjects")
        objects = parse_hit_objects(hit_objects)
        difficulty = parse_difficulty(sections)
        beatmap.add_difficulty(diff, objects, difficulty)
    return beatmap