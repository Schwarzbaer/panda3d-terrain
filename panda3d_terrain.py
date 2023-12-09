# "Fast Hydraulic Erosion Simulation and Visualization on GPU": https://xing-mei.github.io/files/erosion.pdf
# Model Parameters
# * `g`: Gravity
# * `l`: Length of the pipe connecting cells
# * `A`: Cross-section of the pipe connecting cells
# * `Ks`: Dissolving constant
# * `Kd`: Deposition constant
# * `Ke`: Evaporation constant
# Frame Parameters
# * `dt`: time step.
# Map Data
# * `w`: Water influx rate during this frame
# * `b`: Terrain height
# * `d`: Water height
# * `s`: Suspended sediment amount
# * `f`: Outflow flux
# * `v`: Velocity field
# Intermediate Maps
# * `d1`: post-increase water height
# Process
# * DONE: Increase water
#   Apply desired changes to `w`
#   d1 = d + w * dt
# * Compute outflux flow; We treat neighboring cells as if they were connected by pipes.
#   Consider boundary conditions! Make flux to edges 0, or loop it around.
#   * `cf`: Current flux
#   * `dh`: Height difference between cells = (own terrain height + own `d1`) - (neighbor terrain height + neighbor `d1`)
#   pipe coefficient = A * (g * dh / l)
#   neighbor flux = max(0, current flux + dt * pipe coefficient)
#   scaling factor = min(1, `d1` * xy distance / (sum of all flux * dt))
#   total flux = scaling factor * all neighbor fluxes
# * Update water surface
#   net water volume change = dt * all neighbor fluxes
#   new water height `d2` = `d1` + net water volume change / distance to neighors  # WTF???
# * Update velocity field
#   Consider only vertical flow.
#   `u` / `v` are velocity in X / Y direction
#   For X and Y direction separately:
#   * delta Water `dWX` = sum up the delta flows with the neighbors.
#   * equation: `dWX` = distance Y (???) * (average of `d1` and `d2`) * u
#   * derive u (and v accordingly for `dWY`).
#   For simulation to be stable, dt * u/v velocity <= distance to neighbor
# * Erosion-deposition
#   Note: local tilt angle may require setting a lower bound, otherwise sediment transport capacity will be near zero at low tilt.
#   sediment transport capacity `C` = scaling factor * sin(local tilt angle) * |velocity|
#   if sediment transport capacity > suspended sediment amount:  # add soil to water
#       dissolved amount = Ks * (C - s)
#       new terrain height = b - dissolved amount
#       intermediate sediment amount `s1` = s + dissolved amount
#   else:  # deposit sediment
#       deposited amount = Kd * (s - C)
#       new terrain height = b + deposited amount
#       intermediate sediment amount `s1` = s - deposited amount
# * Transport sediment
#   new suspended sediment amount = s1 at position - uv * dt, interpolating the four nearest neighbors
# * DONE: Evaporate water
#   Temperature is assumed to be the same everywhere.
#   new water height = `d2` + (1 - Ke * dt)

from panda3d.core import load_prc_file_data
from panda3d.core import SamplerState
from panda3d.core import PfmFile
from panda3d.core import Texture
from panda3d.core import NodePath
from panda3d.core import PerlinNoise2
from panda3d.core import Vec4
from panda3d.core import Shader
from panda3d.core import GeomVertexFormat
from panda3d.core import GeomVertexData
from panda3d.core import GeomVertexWriter
from panda3d.core import InternalName
from panda3d.core import GeomTriangles
from panda3d.core import Geom
from panda3d.core import GeomNode
from panda3d.core import ComputeNode

from direct.showbase.ShowBase import ShowBase

load_prc_file_data('', 'gl-version 3 2')

resolution = 32
f_res = float(resolution)

evaporation_constant = 0.0

ShowBase()
base.disable_mouse()
base.accept('escape', base.task_mgr.stop)


vertex_shader = """
#version 430

in vec4 vertex;
in vec2 texcoord;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform sampler2D terrainHeight;
uniform sampler2D waterHeight;

out vec2 uv;
out float localTerrainHeight;
out float localWaterHeight;

void main()  {
  uv = texcoord;
  localTerrainHeight = texture2D(terrainHeight, texcoord).x;
  localWaterHeight = texture2D(waterHeight, texcoord).x;
  vec4 finalPos = vertex;
  finalPos.z = localTerrainHeight + localWaterHeight;
  gl_Position = p3d_ModelViewProjectionMatrix * finalPos;
}
"""
fragment_shader = """
#version 430

in vec2 uv;
in float localTerrainHeight;
in float localWaterHeight;

layout(location = 0) out vec4 diffuseColor;

vec4 green = vec4(0.0, 1.0, 0.0, 1.0);
vec4 gray = vec4(0.5, 0.5, 0.5, 1.0);
vec4 blue = vec4(0.0, 0.0, 1.0, 1.0);

void main () {
  vec4 groundColor = mix(green, gray, localTerrainHeight);
  float waterFactor = min(1.0, localWaterHeight * 20.0);
  diffuseColor = mix(groundColor, blue, waterFactor);
}
"""

def make_model():
    v_format = GeomVertexFormat.get_v3t2()

    v_data = GeomVertexData("Data", v_format, Geom.UHStatic)
    v_data.unclean_set_num_rows(resolution ** 2)
    vertex = GeomVertexWriter(v_data, InternalName.get_vertex())
    texcoord = GeomVertexWriter(v_data, InternalName.get_texcoord())

    tris = GeomTriangles(Geom.UHStatic)
    for x in range(resolution):
        x_f = float(x)
        for y in range(resolution):
            y_f = float(y)
            vertex.set_data3f(x_f / (resolution - 1), y_f / (resolution - 1), 0)
            texcoord.set_data2f(x_f / (resolution - 1), y_f / (resolution - 1))
    for x in range(resolution - 1):
        for y in range(resolution - 1):
            bottom_left = y * resolution + x
            bottom_right = y * resolution + x + 1
            top_left = (y + 1) * resolution + x
            top_right = (y + 1) * resolution + x + 1
            tris.add_vertices(bottom_right, bottom_left, top_left)
            tris.add_vertices(top_left, top_right, bottom_right)
    tris.close_primitive()

    geom = Geom(v_data)
    geom.add_primitive(tris)
    node = GeomNode('geom_node')
    node.add_geom(geom)
    surface = NodePath(node)
    return surface


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

# Putting the visual terrain together...
visual_terrain_np = make_model()
visual_terrain_shader = Shader.make(
    Shader.SL_GLSL,
    vertex=vertex_shader,
    fragment=fragment_shader,
)
visual_terrain_np.set_shader(visual_terrain_shader)
visual_terrain_np.set_shader_input("terrainHeight", terrain_height)
visual_terrain_np.set_shader_input("waterHeight", water_height)

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
    compute_np.reparent_to(visual_terrain_np)
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
compute_node_add_water = add_compute_node(
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
compute_node_evaporate = add_compute_node(
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
add_compute_node(
    update_main_data,
    2,
    dict(
        waterHeightAfterEvaporation=water_height_after_evaporation,
        waterHeight=water_height,
    ),
)

# Attaching the terrain to the scene
visual_terrain_np.reparent_to(base.render)
visual_terrain_np.set_pos(-1.0, -1.0, 0.0)
visual_terrain_np.set_sx(2.0)
visual_terrain_np.set_sy(2.0)

base.cam.set_pos(2, -2, 2)
base.cam.look_at(0, 0, 0.25)


def end_of_influx():
    compute_node_evaporate.set_shader_input("evaporationConstant", 1.0)
    water_influx_img.fill((0.0, 0, 0, 0))
    water_influx.load(water_influx_img)
base.accept("space", end_of_influx)

base.run()
