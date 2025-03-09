def nth_bit_set(n, i):
    return (n & (1 << i)) != 0

def osu_to_screen(osu_x, osu_y, screen_width = 1920, screen_height = 1080):

    OSU_PLAYFIELD_WIDTH = 512
    OSU_PLAYFIELD_HEIGHT = 384

    scaling_factor = screen_height / OSU_PLAYFIELD_HEIGHT

    scaled_playfield_width = OSU_PLAYFIELD_WIDTH * scaling_factor

    horizontal_offset = (screen_width - scaled_playfield_width) / 2

    screen_x = osu_x * scaling_factor + horizontal_offset
    screen_y = osu_y * scaling_factor  

    return (screen_x, screen_y)
