from panda3d.core import NodePath
from panda3d.core import PfmFile
from panda3d.core import Texture
from panda3d.core import Shader
from panda3d.core import PerlinNoise2
from panda3d.core import ComputeNode


resolution = 128*4
f_res = float(resolution)


evaporation_constant = 0.3
pipe_coefficient = 9.81 * 10.0  # gravity g * pipe cross area A / pipe length l
cell_distance = 1.0


def make_map(name, channels=1):
    assert channels in [1, 4]
    image = PfmFile()
    image.clear(
        x_size=resolution,
        y_size=resolution,
        num_channels=channels,
    )
    if channels == 1:
        image.fill(0.0)
    else:
        image.fill((0.0, 0.0, 0.0, 0.0))

    texture = Texture(name)
    texture.load(image)
    if channels == 1:
        texture.set_format(Texture.F_r16)
    else:
        texture.set_format(Texture.F_rgba16)
    texture.wrap_u = Texture.WM_clamp
    texture.wrap_v = Texture.WM_clamp

    return texture, image


terrain_height,                 terrain_height_img = make_map('terrain_height')
water_height,                   water_height_img   = make_map('water_height')
water_influx,                   water_influx_img   = make_map('water_influx')
water_height_after_influx,      _                  = make_map('water_height_after_influx')
water_crossflux,                _                  = make_map('water_crossflux', channels=4)
water_height_after_crossflux,   _                  = make_map('water_height_after_crossflux')
water_height_after_evaporation, _                  = make_map('water_height_after_evaporation')

mem_use = sum(
    [t.estimate_texture_memory() for t in
     [
         terrain_height,
         water_height,
         water_influx,
         water_height_after_influx,
         water_crossflux,
         water_height_after_crossflux,
         water_height_after_evaporation,
     ]
     ]
)
print(f"Memory use: {mem_use} bytes")

# Terrain height from Perlin noise
noise_generator = PerlinNoise2() 
noise_generator.setScale(0.2)
for x in range(resolution):
    coord_x = x / f_res
    for y in range(resolution):
        coord_y = y / f_res
        local_height = noise_generator.noise(coord_x, coord_y) / 2.0 + 0.5
        local_height *= 0.5
        terrain_height_img.set_point1(x, y, local_height)
terrain_height.load(terrain_height_img)
terrain_height.set_format(Texture.F_r16)
#terrain_height.wrap_u = Texture.WM_clamp
#terrain_height.wrap_v = Texture.WM_clamp

# Water at the beginning
for x in range(3 * resolution//8, 5 * resolution//8):
    coord_x = x / f_res
    for y in range(3 * resolution//8, 5 * resolution//8):
        coord_y = y / f_res
        water_height_img.set_point4(x, y, (0.5, 0, 0, 0))
water_height.load(water_height_img)
water_height.set_format(Texture.F_r16)
#water_height.wrap_u = Texture.WM_clamp
#water_height.wrap_v = Texture.WM_clamp


# Small global water influx
# for x in range(resolution):
#     coord_x = x / f_res
#     for y in range(resolution):
#         coord_y = y / f_res
#water_influx_img.set_point1(resolution//2, resolution//2, 10.0)
#water_influx.load(water_influx_img)
#water_influx.set_format(Texture.F_r16)


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
layout(r16f) uniform readonly image2D waterHeight;
layout(r16f) uniform readonly image2D waterInflux;
layout(r16f) uniform writeonly image2D waterHeightAfterInflux;

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


calculate_outflux = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
uniform float pipeCoefficient;
uniform float cellDistance;
layout(r16f) uniform readonly image2D terrainHeight;
layout(r16f) uniform readonly image2D waterHeightAfterInflux;
layout(rgba16f) uniform image2D waterCrossflux;

const ivec2 deltaCoord[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  float localWaterHeight = imageLoad(waterHeightAfterInflux, coord).x;
  float localTotalHeight = imageLoad(terrainHeight, coord).x + localWaterHeight;
  vec4 currentCrossflux = imageLoad(waterCrossflux, coord);
  const int arraySize = 4;
  float crossflux[arraySize] = float[](0.0, 0.0, 0.0, 0.0);
  crossflux[0] = currentCrossflux.r;
  crossflux[1] = currentCrossflux.g;
  crossflux[2] = currentCrossflux.b;
  crossflux[3] = currentCrossflux.a;
  for (int i = 0; i < 4; i++) {
    ivec2 neighborCoord = coord + deltaCoord[i];
    float neighborTotalHeight = imageLoad(terrainHeight, neighborCoord).x + imageLoad(waterHeightAfterInflux, neighborCoord).x;
    float deltaHeight = localTotalHeight - neighborTotalHeight;
    crossflux[i] = max(0.0, crossflux[i] + deltaHeight * pipeCoefficient * dt);
  }
  float scalingFactor = min(1, localWaterHeight * cellDistance * cellDistance / ((crossflux[0] + crossflux[1] + crossflux[2] + crossflux[3]) * dt));
  for (int i = 0; i < 4; i++) {
    crossflux[i] *= scalingFactor;
  }
  imageStore(waterCrossflux, coord, vec4(crossflux[0], crossflux[1], crossflux[2], crossflux[3]));
}
"""
calculate_outflux_cn = add_compute_node(
    calculate_outflux,
    1,
    dict(
        dt=1.0/60.0,
        pipeCoefficient=pipe_coefficient,
        cellDistance=cell_distance,
        terrainHeight=terrain_height,
        waterHeightAfterInflux=water_height_after_influx,
        waterCrossflux=water_crossflux,
    ),
)


apply_crossflux = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
uniform float cellDistance;
layout(r16f) uniform readonly image2D waterHeightAfterInflux;
layout(rgba16f) uniform readonly image2D waterCrossflux;
layout(r16f) uniform writeonly image2D waterHeightAfterCrossflux;

const ivec2 deltaCoord[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  float inflowLeft = imageLoad(waterCrossflux, coord + deltaCoord[0]).g;
  float inflowRight = imageLoad(waterCrossflux, coord + deltaCoord[1]).r;
  float inflowBottom = imageLoad(waterCrossflux, coord + deltaCoord[2]).a;
  float inflowTop = imageLoad(waterCrossflux, coord + deltaCoord[3]).b;
  float totalInflow = inflowLeft + inflowRight + inflowBottom + inflowTop;
  vec4 outflows = imageLoad(waterCrossflux, coord);
  float totalOutflow = outflows.r + outflows.g + outflows.b + outflows.a;
  float deltaVolume = (totalInflow - totalOutflow) * dt;
  float oldWaterHeight = imageLoad(waterHeightAfterInflux, coord).x;
  float newWaterHeight = oldWaterHeight + deltaVolume / (cellDistance * cellDistance);
  imageStore(waterHeightAfterCrossflux, coord, vec4(newWaterHeight, 0.0, 0.0, 0.0));
}
"""
apply_crossflux_cn = add_compute_node(
    apply_crossflux,
    1,
    dict(
        dt=1.0/60.0,
        cellDistance=cell_distance,
        waterHeightAfterInflux=water_height_after_influx,
        waterCrossflux=water_crossflux,
        waterHeightAfterCrossflux=water_height_after_crossflux,
    ),
)


evaporate = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float evaporationConstant;
uniform float dt;
layout(r16f) uniform readonly image2D waterHeightAfterCrossflux;
layout(r16f) uniform writeonly image2D waterHeightAfterEvaporation;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newWaterHeight = imageLoad(waterHeightAfterCrossflux, coord) * (1.0 - evaporationConstant * dt);
  imageStore(waterHeightAfterEvaporation, coord, newWaterHeight);
}
"""
evaporate_cn = add_compute_node(
    evaporate,
    2,
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
layout(r16f) uniform readonly image2D waterHeightAfterEvaporation;
layout(r16f) uniform writeonly image2D waterHeight;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newWaterHeight = imageLoad(waterHeightAfterEvaporation, coord);
  imageStore(waterHeight, coord, newWaterHeight);
}
"""
update_main_data_cn = add_compute_node(
    update_main_data,
    3,
    dict(
        waterHeightAfterEvaporation=water_height_after_evaporation,
        waterHeight=water_height,
    ),
)


compute_nodes = [
    add_water_cn,
    calculate_outflux_cn,
    apply_crossflux_cn,
    evaporate_cn,
    update_main_data_cn,
]
