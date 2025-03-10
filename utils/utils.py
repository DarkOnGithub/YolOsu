import re

def nth_bit_set(n, i):
    return (n & (1 << i)) != 0
def osu_pixels_to_normal_coords(osu_x, osu_y, resolution_width, resolution_height):

    osu_y -= 8
    playfield_height = 0.8 * resolution_height
    playfield_width = (4/3) * playfield_height 
    
    visible_aspect_ratio = resolution_width / resolution_height
    
    playfield_left = (resolution_width - playfield_width) / 2
    playfield_top = (resolution_height - playfield_height) / 2 + (0.017 * resolution_height)
    
    scale = playfield_height / 384
    
    screen_x = playfield_left + (osu_x * scale)
    screen_y = playfield_top + (osu_y * scale)
    
    return screen_x, screen_y 

def get_difficulty(name):
    match = re.search(r"\[(.*?)\]", name)
    if match:
        group_in_brackets = match.group(1)
        return group_in_brackets
    else:
        return None