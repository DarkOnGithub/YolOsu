import zipfile
from typing import Dict, List, Optional, Tuple
from emulator.objects import HitCircle, Slider
from emulator import Beatmap
import utils.utils as utils
import re
import os
import bisect

class TimingPoint:
    def __init__(self, offset: float, ms_per_beat: float, uninherited: bool):
        self.offset = offset
        self.ms_per_beat = ms_per_beat
        self.uninherited = uninherited

class TimingParser:
    def __init__(self, timing_points: List[TimingPoint]):
        self.uninherited_points = sorted(
            [tp for tp in timing_points if tp.uninherited],
            key=lambda x: x.offset
        )
        self.offsets = [tp.offset for tp in self.uninherited_points]
        
    def get_beat_duration(self, time: float) -> float:
        if not self.uninherited_points:
            return 600.0  
        
        idx = bisect.bisect_right(self.offsets, time) - 1
        return self.uninherited_points[idx].ms_per_beat if idx >= 0 else 600.0

def extract_osu_file(osz_path: str) -> Dict[str, str]:
    difficulties = {}
    
    base_name = os.path.basename(osz_path).split('.')[0]
    extract_dir = os.path.join(os.path.dirname(osz_path), base_name)
    
    try:
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            with zipfile.ZipFile(osz_path, "r") as z:
                z.extractall(extract_dir)
        
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".osu"):
                    file_path = os.path.join(root, file)
                    difficulty = utils.get_difficulty(file)
                    if difficulty is None:
                        print(f"Could not find difficulty for {file}")
                        continue
                    with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        content = f.read()
                        difficulties[difficulty] = content
        
        if not difficulties:
            print(f"No .osu files found in extracted directory: {extract_dir}")
            
    except zipfile.BadZipFile:
        print(f"Error: Invalid .osz file at {osz_path}")
    except Exception as e:
        print(f"Error extracting/reading files: {e}")        
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

def parse_hit_objects(hit_objects: List[str], timing_parser: TimingParser) -> List:
    objects = []
    for line in hit_objects:
        parts = line.split(',')
        x, y, time, type_ = map(int, parts[:4])
        
        if type_ & 2:  
            params = parts[5].split('|')
            curve_type = params[0][0]
            curve_points = [tuple(map(int, p.split(':'))) for p in params[1:]]
            length = float(parts[7])
            beat_duration = timing_parser.get_beat_duration(time)
            objects.append(Slider(x, y, time, curve_type, curve_points, length, beat_duration))
        elif type_ & 1:
            objects.append(HitCircle(x, y, time))
    return objects

def parse_timing_points(timing_lines: List[str]) -> List[TimingPoint]:
    points = []
    for line in timing_lines:
        parts = line.split(',')
        if len(parts) < 8:
            continue
            
        offset = float(parts[0])
        ms_per_beat = float(parts[1])
        uninherited = int(parts[6]) == 1
        
        points.append(TimingPoint(offset, ms_per_beat, uninherited))
    return points

def parse_difficulty(sections: Dict[str, List[str]]) -> Optional[List[float]]:
    difficulty_section = sections.get("Difficulty")
    if not difficulty_section:
        return None
        
    pattern = re.compile(r":([\d.]+)")
    return [float(match.group(1)) 
            for match in pattern.finditer("\n".join(difficulty_section))]

def parse_osz_file(osz_path: str) -> Beatmap.Beatmap:
    osz_path = "maps/" + osz_path + ".osz"
    filename = os.path.basename(osz_path) 
    match = re.match(r"(\d+)\s+(.+?)\.osz$", filename)
        
    if match is None:
        print(f"Error: Invalid beatmap filename {filename}")
        return
    beatmap = Beatmap.Beatmap(int(match.group(1)), match.group(2).split(" - ", 1)[-1])
    difficulties = extract_osu_file(osz_path)
    
    if not difficulties:
        return beatmap
    print(f"Extracted difficulties: {list(difficulties.keys())}")
    for diff, content in difficulties.items():
        sections = parse_osu_file(content)
        if sections:
            timing_points = sections.get("TimingPoints", [])
            timing_parser = TimingParser(parse_timing_points(timing_points))
            objects = parse_hit_objects(sections["HitObjects"], timing_parser)
            difficulty = parse_difficulty(sections)
            beatmap.add_difficulty(diff, objects, difficulty, timing_parser)
            
    return beatmap