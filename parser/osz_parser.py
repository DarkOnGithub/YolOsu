import zipfile
from typing import Dict, List, Optional, Tuple
from emulator.objects import HitCircle, Slider
from emulator import Beatmap
import utils.utils as utils
import re

def extract_osu_file(osz_path: str) -> Dict[str, str]:

    difficulties = {}
    try:
        with zipfile.ZipFile(osz_path, "r") as z:
            
            osu_files = (f for f in z.namelist() if f.lower().endswith(".osu"))
            for file in osu_files:
                with z.open(file) as f:
                    difficulty = utils.get_difficulty(file)
                    if difficulty is None:
                        print(f"Could not find difficulty for {file}")
                        continue
                    
                    content = f.read().decode('utf-8-sig', errors='replace')
                    difficulties[difficulty] = content
    except zipfile.BadZipFile:
        print(f"Error: Invalid .osz file at {osz_path}")
    return difficulties

def parse_osu_file(osu_content: str) -> Optional[Dict[str, List[str]]]:

    if not osu_content:
        return None
        
    sections: Dict[str, List[str]] = {}
    current_section = None
    
    
    
    lines = (line.strip() for line in osu_content.splitlines() if line.strip())
    
    for line in lines:
        if line[0] == '[' and line[-1] == ']':
            current_section = line[1:-1]
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)
    
    return sections

def parse_hit_objects(hit_objects: List[str]) -> List:
    objects = []
    
    
    curve_split = re.compile(r":")
    
    for i, line in enumerate(hit_objects):
        parts = line.split(",")
        x, y, time, type_ = map(int, parts[:4])
        params = parts[5:]
        
        
        is_slider = type_ & 2  
        is_circle = type_ & 1  
        
        if is_circle:
            objects.append(HitCircle(x, y, time))
            
        if is_slider and params:
            curve_info = params[0].split("|")
            curve_type = curve_info[0]
            
            curve_points = [tuple(map(int, curve_split.split(p))) 
                          for p in curve_info[1:]]
            length = int(params[1])
            objects.append(Slider(x, y, time, curve_type, curve_points, length))
            
    return objects

def parse_difficulty(sections: Dict[str, List[str]]) -> Optional[List[float]]:
    difficulty_section = sections.get("Difficulty")
    if not difficulty_section:
        return None
        
    
    pattern = re.compile(r":([\d.]+)")
    return [float(match.group(1)) 
            for match in pattern.finditer("\n".join(difficulty_section))]

def parse_osz_file(osz_path: str) -> Beatmap.Beatmap:

    beatmap = Beatmap.Beatmap()
    difficulties = extract_osu_file(osz_path)
    
    if not difficulties:
        return beatmap
    print(f"Extracted difficulties: {list(difficulties.keys())}")
    for diff, content in difficulties.items():
        print(f"Parsing difficulty: {diff}")
        sections = parse_osu_file(content)
        if sections and "HitObjects" in sections:
            objects = parse_hit_objects(sections["HitObjects"])
            difficulty = parse_difficulty(sections)
            beatmap.add_difficulty(diff, objects, difficulty)
            
    return beatmap