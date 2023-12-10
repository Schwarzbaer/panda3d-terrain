from panda3d.core import load_prc_file_data
from panda3d.core import NodePath
from panda3d.core import Shader
from panda3d.core import GeomVertexFormat
from panda3d.core import GeomVertexData
from panda3d.core import GeomVertexWriter
from panda3d.core import InternalName
from panda3d.core import GeomTriangles
from panda3d.core import Geom
from panda3d.core import GeomNode

from direct.showbase.ShowBase import ShowBase

import simulator
resolution = simulator.resolution
f_res = simulator.f_res


load_prc_file_data('', 'gl-version 3 2')
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


# Putting the visual terrain together...
visual_terrain_np = make_model()
visual_terrain_shader = Shader.make(
    Shader.SL_GLSL,
    vertex=vertex_shader,
    fragment=fragment_shader,
)
visual_terrain_np.set_shader(visual_terrain_shader)
visual_terrain_np.set_shader_input("terrainHeight", simulator.terrain_height)
visual_terrain_np.set_shader_input("waterHeight", simulator.water_height)


# Attaching the terrain to the scene
visual_terrain_np.reparent_to(base.render)
visual_terrain_np.set_pos(-1.0, -1.0, 0.0)
visual_terrain_np.set_sx(2.0)
visual_terrain_np.set_sy(2.0)


# ...and attaching the compute nodes
for node in simulator.compute_nodes:
    node.reparent_to(visual_terrain_np)


# Camera needs some love, too
base.cam.set_pos(2, -2, 2)
base.cam.look_at(0, 0, 0.25)

# And we want to stop the influx of water, and begin evaporation.
def end_of_influx():
    simulator.evaporate_cn.set_shader_input("evaporationConstant", 1.0)
    simulator.water_influx_img.fill((0.0, 0, 0, 0))
    simulator.water_influx.load(simulator.water_influx_img)
base.accept("space", end_of_influx)

base.run()
