import re

def nth_bit_set(n, i):
    return (n & (1 << i)) != 0

def osu_pixels_to_normal_coords(osu_x, osu_y, resolution_width, resolution_height):

    playfield_height = 0.8 * resolution_height
    playfield_width = (4/3) * playfield_height

    playfield_left = (resolution_width - playfield_width) / 2
    playfield_top = (resolution_height - playfield_height) / 2 + (0.02 * playfield_height)

    osu_scale = playfield_height / 384 

    mapped_x = (osu_x * osu_scale) + playfield_left
    mapped_y = (osu_y * osu_scale) + playfield_top

    return mapped_x, mapped_y

def get_difficulty(name):
    match = re.search(r"\[(.*?)\]", name)

    if match:
        group_in_brackets = match.group(1)
        words = group_in_brackets.split()
        difficulty = words[-1]
        return difficulty
    else:
        return None