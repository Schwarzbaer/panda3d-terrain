from panda3d.core import NodePath
from panda3d.core import Shader
from panda3d.core import GeomVertexFormat
from panda3d.core import GeomVertexData
from panda3d.core import GeomVertexWriter
from panda3d.core import InternalName
from panda3d.core import GeomTriangles
from panda3d.core import Geom
from panda3d.core import GeomNode


double_heightmap_shader = """
#version 430

in vec4 vertex;
in vec2 texcoord;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform sampler2D heightA;
uniform sampler2D heightB;
uniform sampler2D normals;

out vec2 uv;
out float height;
out vec3 normal;

void main()  {
  uv = texcoord;
  float localHeightA = texture(heightA, texcoord).x;
  float localHeightB = texture(heightB, texcoord).x;
  height = localHeightA + localHeightB - 0.001;
  vec4 finalPos = vertex;
  finalPos.z = height;
  gl_Position = p3d_ModelViewProjectionMatrix * finalPos;
  normal = texture(normals, texcoord).xyz * 2.0 - 1.0;
}
"""
heightmap_shader = """
#version 430

in vec4 vertex;
in vec2 texcoord;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform sampler2D heightA;
//uniform sampler2D normals;

out vec2 uv;
out float height;
//out vec4 normal;

void main()  {
  uv = texcoord;
  height = texture(heightA, texcoord).x;
  vec4 finalPos = vertex;
  finalPos.z = height;
  gl_Position = p3d_ModelViewProjectionMatrix * finalPos;
  //normal = texture(normals, texcoord);
}
"""
terrain_shader = """
#version 430

in vec2 uv;
in float height;
//in vec4 normal;

layout(location = 0) out vec4 diffuseColor;

vec4 green = vec4(0.0, 1.0, 0.0, 1.0);
vec4 gray = vec4(0.5, 0.5, 0.5, 1.0);

void main () {
  diffuseColor = mix(green, gray, height*2.0);
}
"""
water_shader = """
#version 430

in vec2 uv;
in float height;
in vec3 normal;

layout(location = 0) out vec4 diffuseColor;

vec3 light_direction = normalize(vec3(0.2, 0.2, 1.0));
vec3 lightColor = vec3(1.0, 1.0, 1.0);
vec3 waterColor = vec3(0.5, 0.5, 1.0);

void main () {
  float lambertian_diffusion_weight = max(0.0, dot(light_direction, normal));
  diffuseColor = vec4(waterColor * lightColor * lambertian_diffusion_weight, 1.0);
}
"""


def make_model(resolution):
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


def make_terrain(simulator, resolution=None):
    if resolution is None:
        resolution = simulator.resolution
    # Putting the visual terrain together...
    visual_terrain_np = make_model(resolution)
    visual_terrain_shader = Shader.make(
        Shader.SL_GLSL,
        vertex=heightmap_shader,
        fragment=terrain_shader,
    )
    visual_terrain_shader.set_filename(Shader.ST_none, 'terrain')
    visual_terrain_np.set_shader(visual_terrain_shader)
    visual_terrain_np.set_shader_input("heightA", simulator.textures['terrain_height'])
    #visual_terrain_np.set_shader_input("normals", simulator.textures['water_normal_map'])
    
    
    visual_water_np = make_model(resolution)
    visual_water_shader = Shader.make(
        Shader.SL_GLSL,
        vertex=double_heightmap_shader,
        fragment=water_shader,
    )
    visual_water_shader.set_filename(Shader.ST_none, 'water')
    visual_water_np.set_shader(visual_water_shader)
    visual_water_np.set_shader_input("heightA", simulator.textures['terrain_height'])
    visual_water_np.set_shader_input("heightB", simulator.textures['water_height'])
    visual_water_np.set_shader_input("normals", simulator.textures['water_normal_map'])
    
    
    visual_water_np.reparent_to(visual_terrain_np)
    
    
    # Attaching the terrain to the scene
    visual_terrain_np.set_pos(-1.0, -1.0, 0.0)
    visual_terrain_np.set_sx(2.0)
    visual_terrain_np.set_sy(2.0)

    return visual_terrain_np
