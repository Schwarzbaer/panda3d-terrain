from math import pi
from math import cos

from panda3d.core import PerlinNoise2
from panda3d.core import StackedPerlinNoise2


def fill_data(image, func):
    x_size = image.get_x_size()
    y_size = image.get_y_size()
    for x in range(x_size):
        coord_x = x / float(x_size)
        for y in range(y_size):
            coord_y = y / float(y_size)
            image.set_point1(x, y, func(coord_x, coord_y))


def funnel(image):
    def height(x, y):
        return max(abs(x - 0.5), abs(y - 0.5))
    fill_data(image, height)


def sine_hills(image, phases=2):
    def height(x, y):
        height_1 = cos((x - 0.5) * pi * (2 + (phases - 1) * 4)) * 0.25
        height_2 = cos((y - 0.5) * pi * (2 + (phases - 1) * 4)) * 0.25
        return max(0.0, height_1 + height_2)
    fill_data(image, height)


def perlin(image):
    noise_generator = PerlinNoise2() 
    noise_generator.setScale(0.2)
    def height(x, y):
        return (noise_generator.noise(x, y) / 2.0 + 0.5) * 0.5 + 0.5
    fill_data(image, height)


def perlin_terrain(image, min_height, max_height, base_freq=1.0):
    noise_generator = StackedPerlinNoise2(base_freq, base_freq, 4, 2.0, 0.4)
    height_diff = max_height - min_height
    def height(x, y):
        local_height = noise_generator.noise(x, y) * 0.5 + 0.5  # Values in 0.0 - 1.0
        local_height = local_height * height_diff + min_height  # scaled to given range
        return local_height
    fill_data(image, height)


def half(image):
    image.fill(0.25)


def block(image, phases=2):
    def height(x, y):
        if (0.33 < x < 0.66) and (0.33 < y < 0.66):
            return 0.5
        else:
            return 0.0
    fill_data(image, height)
