from panda3d.core import NodePath
from panda3d.core import PfmFile
from panda3d.core import Texture
from panda3d.core import Shader
from panda3d.core import PerlinNoise2
from panda3d.core import ComputeNode


resolution = 32
f_res = float(resolution)


evaporation_constant = 0.0


def make_map():
    image = PfmFile()
    image.clear(
        x_size=resolution,
        y_size=resolution,
        num_channels=4,
    )
    image.fill((0.0, 0.0, 0.0, 1.0))

    texture = Texture('')
    texture.setup_2d_texture(
        resolution,
        resolution,
        Texture.T_float,
        Texture.F_rgba32,
    )
    texture.wrap_u = Texture.WM_clamp
    texture.wrap_v = Texture.WM_clamp

    texture.load(image)
    return texture, image


terrain_height,                 terrain_height_img = make_map()
water_height,                   water_height_img   = make_map()
water_influx,                   water_influx_img   = make_map()
water_crossflux,                _                  = make_map()
water_height_after_influx,      _                  = make_map()
water_height_after_crossflux,   _                  = water_height_after_influx, None  #make_map()
water_height_after_evaporation, _                  = make_map()

# Terrain height from Perlin noise
noise_generator = PerlinNoise2() 
noise_generator.setScale(0.1) 
for x in range(resolution):
    coord_x = x / f_res
    for y in range(resolution):
        coord_y = y / f_res
        local_height = noise_generator.noise(coord_x, coord_y) / 2.0 + 0.5
        local_height *= 0.5
        terrain_height_img.set_point4(x, y, (local_height, 0, 0, 1))
terrain_height.load(terrain_height_img)

# Water at the beginning
for x in range(resolution//2):
    coord_x = x / f_res
    for y in range(resolution//2):
        coord_y = y / f_res
        water_height_img.set_point4(x, y, (0.2, 0, 0, 0))
water_height.load(water_height_img)

# Small global water influx
for x in range(resolution//2):
    coord_x = x / f_res
    for y in range(resolution//2):
        coord_y = y / f_res
        water_influx_img.set_point4(x, y, (0.01, 0, 0, 0))
water_influx.load(water_influx_img)


def add_compute_node(code, cull_bin_sort, inputs):
    compute_shader = Shader.make_compute(
        Shader.SL_GLSL,
        code,
    )
    workgroups = (resolution // 16, resolution // 16, 1)
    compute_node = ComputeNode("compute")
    compute_node.add_dispatch(*workgroups)
    compute_np = NodePath(compute_node)
    compute_np.set_shader(compute_shader)
    for glsl_name, value in inputs.items():
        compute_np.set_shader_input(glsl_name, value)
    compute_np.setBin("fixed", cull_bin_sort)
    return compute_np


add_water = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
layout(rgba32f) uniform readonly image2D waterHeight;
layout(rgba32f) uniform readonly image2D waterInflux;
layout(rgba32f) uniform writeonly image2D waterHeightAfterInflux;


void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newWaterHeight = imageLoad(waterHeight, coord) + imageLoad(waterInflux, coord) * dt;
  imageStore(waterHeightAfterInflux, coord, newWaterHeight);
}
"""
add_water_cn = add_compute_node(
    add_water,
    0,
    dict(
        dt=1.0/60.0,
        waterHeight=water_height,
        waterInflux=water_influx,
        waterHeightAfterInflux=water_height_after_influx,
    ),
)
evaporate = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float evaporationConstant;
uniform float dt;
layout(rgba32f) uniform readonly image2D waterHeightAfterCrossflux;
layout(rgba32f) uniform writeonly image2D waterHeightAfterEvaporation;


void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newWaterHeight = imageLoad(waterHeightAfterCrossflux, coord) * (1.0 - evaporationConstant * dt);
  imageStore(waterHeightAfterEvaporation, coord, newWaterHeight);
}
"""
evaporate_cn = add_compute_node(
    evaporate,
    1,
    dict(
        dt=1.0/60.0,
        evaporationConstant=evaporation_constant,
        waterHeightAfterCrossflux=water_height_after_crossflux,
        waterHeightAfterEvaporation=water_height_after_evaporation,
    ),
)


update_main_data = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
layout(rgba32f) uniform readonly image2D waterHeightAfterEvaporation;
layout(rgba32f) uniform writeonly image2D waterHeight;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newWaterHeight = imageLoad(waterHeightAfterEvaporation, coord);
  imageStore(waterHeight, coord, newWaterHeight);
}
"""
update_main_data_cn = add_compute_node(
    update_main_data,
    2,
    dict(
        waterHeightAfterEvaporation=water_height_after_evaporation,
        waterHeight=water_height,
    ),
)
