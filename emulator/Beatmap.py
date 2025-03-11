from ast import With
import os
import re
import sys
from tkinter import W
import shutil
class Beatmap:
    def __init__(self, id, name):
        print(f"Creating beatmap with id {id} and name {name}")
        self.difficulties = {}
        self.name = name
        self.id = id
    def add_difficulty(self, name, hit_objects, difficulty):
        self.difficulties[name] = Difficulty(name, hit_objects, difficulty)
        
    def generate_clip(self, difficulty):
        if difficulty in self.difficulties:
            return self.difficulties[difficulty].generate_clip(self.name)
        return None
class Difficulty:
    def __init__(self, name, hit_objects, difficulty):
        self.name = name
        self.hit_objects = hit_objects
    
        self.difficulty = {
            "hp": difficulty[0],
            "cs": difficulty[1],
            "od": difficulty[2],
            "ar": difficulty[3]
        }
        
    def generate_clip(self, beatmap_name):
        dirname = sys.path[0]
        file_name = f"{beatmap_name}-{self.name}"
        out_folder = os.path.join(dirname, "replay_output")
        out_path = os.path.join(out_folder, file_name + ".mp4")
        
        danser_path = os.path.join(dirname, "danser")
        if os.path.exists(out_path):
            return out_path
        
        command = (
            f'{danser_path}\\danser-cli.exe'
            f' -t="{beatmap_name}" '
            f' -d="{self.name}" '
            f' -out="{file_name}"'
            f' -record'
            f' -quickstart'
        )
        print(f"Running command: {command}")
        
        if os.system(command) == 0:
            os.makedirs(os.path.dirname(out_folder), exist_ok=True)
            
            source_file = os.path.join(danser_path, "videos", file_name + ".mp4")
            
            try:
                shutil.move(source_file, out_path)
                return out_path
            except Exception as e:
                print(f"Error moving file: {e}")
                if os.path.exists(source_file):
                    return source_file
        
        return None
    
    
    def __str__(self):
        return f"Difficulty {self.name} with {len(self.hit_objects)} hit objects and difficulty {self.difficulty}"