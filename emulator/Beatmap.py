class Beatmap:
    def __init__(self):
        self.difficulties = {}
    
    def add_difficulty(self, name, hit_objects, difficulty):
        self.difficulties[name] = Difficulty(name, hit_objects, difficulty)
        
        
        
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
        
    
    def __str__(self):
        return f"Difficulty {self.name} with {len(self.hit_objects)} hit objects and difficulty {self.difficulty}"