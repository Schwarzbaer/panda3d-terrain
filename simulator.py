import enum

from jinja2 import Template

from panda3d.core import NodePath
from panda3d.core import PfmFile
from panda3d.core import Texture
from panda3d.core import Shader
from panda3d.core import ComputeNode


class BoundaryConditions(enum.Enum):
    OPEN = 1
    CLOSED = 2
    WRAPPING = 3


shader_sources = {}
shader_sources['add_water'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
layout(r16f) uniform readonly image2D heightIn;
layout(r16f) uniform readonly image2D influx;
layout(r16f) uniform writeonly image2D heightOut;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newHeight = imageLoad(heightIn, coord) + imageLoad(influx, coord) * dt;
  imageStore(heightOut, coord, newHeight);
}
"""
shader_sources['calculate_outflux'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
uniform float pipeCoefficient;
uniform float cellDistance;
layout(r16f) uniform readonly image2D terrainHeight;
layout(r16f) uniform readonly image2D waterHeight;
layout(rgba16f) uniform image2D waterCrossflux;

const ivec2 deltaCoord[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

{% if boundary_condition == BoundaryConditions.WRAPPING %}
ivec2 wrapCoord(ivec2 coordIn) {
    return ivec2(mod(coordIn, imageSize(waterCrossflux)));
}
{% else %}
ivec2 wrapCoord(ivec2 coordIn) {
    return ivec2(coordIn);
}
{% endif %}

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 localTerrainHeight = vec4(imageLoad(terrainHeight, coord).x);
  vec4 localWaterHeight = vec4(imageLoad(waterHeight, coord).x);
  vec4 localTotalHeight = localTerrainHeight + localWaterHeight;
  vec4 neighborTerrainHeight = vec4(
    imageLoad(terrainHeight, wrapCoord(coord + ivec2(-1,  0))).x,
    imageLoad(terrainHeight, wrapCoord(coord + ivec2( 1,  0))).x,
    imageLoad(terrainHeight, wrapCoord(coord + ivec2( 0, -1))).x,
    imageLoad(terrainHeight, wrapCoord(coord + ivec2( 0,  1))).x
  );
  vec4 neighborWaterHeight = vec4(
    imageLoad(waterHeight, wrapCoord(coord + ivec2(-1,  0))).x,
    imageLoad(waterHeight, wrapCoord(coord + ivec2( 1,  0))).x,
    imageLoad(waterHeight, wrapCoord(coord + ivec2( 0, -1))).x,
    imageLoad(waterHeight, wrapCoord(coord + ivec2( 0,  1))).x
  );
  vec4 neighborTotalHeight = neighborTerrainHeight + neighborWaterHeight;
  vec4 deltaHeight = localTotalHeight - neighborTotalHeight;
  vec4 crossflux = imageLoad(waterCrossflux, coord);
  crossflux = max(crossflux + deltaHeight * pipeCoefficient * dt, 0.0);
  {% if boundary_condition == BoundaryConditions.CLOSED %}
    // Clamp outflow at boundaries to 0.0
    if (coord.x == 0) {
      crossflux.r = 0.0;
    }
    if (coord.x == imageSize(waterCrossflux).x - 1) {
      crossflux.g = 0.0;
    }
    if (coord.y == 0) {
      crossflux.b = 0.0;
    }
    if (coord.y == imageSize(waterCrossflux).y - 1) {
      crossflux.a = 0.0;
    }
  {% endif %}
  float totalOutflux = crossflux.r + crossflux.g + crossflux.b + crossflux.a;
  float cellArea = cellDistance * cellDistance;
  float waterVolume = localWaterHeight.x * cellArea;
  float scalingFactor = min(waterVolume / (totalOutflux * dt), 1.0);
  crossflux *= scalingFactor;
  imageStore(waterCrossflux, coord, crossflux);
}
"""
shader_sources['apply_crossflux'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
uniform float cellDistance;
layout(r16f) uniform readonly image2D heightIn;
layout(rgba16f) uniform readonly image2D waterCrossflux;
layout(r16f) uniform writeonly image2D heightOut;

const ivec2 deltaCoord[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

{% if boundary_condition == BoundaryConditions.WRAPPING %}
ivec2 wrapCoord(ivec2 coordIn) {
    return ivec2(mod(coordIn, imageSize(waterCrossflux)));
}
{% else %}
ivec2 wrapCoord(ivec2 coordIn) {
    return ivec2(coordIn);
}
{% endif %}

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  float inflowLeft   = imageLoad(waterCrossflux, wrapCoord(coord + deltaCoord[0])).g;
  float inflowRight  = imageLoad(waterCrossflux, wrapCoord(coord + deltaCoord[1])).r;
  float inflowBottom = imageLoad(waterCrossflux, wrapCoord(coord + deltaCoord[2])).a;
  float inflowTop    = imageLoad(waterCrossflux, wrapCoord(coord + deltaCoord[3])).b;
  float totalInflow = inflowLeft + inflowRight + inflowBottom + inflowTop;
  vec4 outflows = imageLoad(waterCrossflux, coord);
  float totalOutflow = outflows.r + outflows.g + outflows.b + outflows.a;
  float deltaVolume = (totalInflow - totalOutflow) * dt;
  float oldWaterHeight = imageLoad(heightIn, coord).x;
  float newWaterHeight = oldWaterHeight + deltaVolume / (cellDistance * cellDistance);
  imageStore(heightOut, coord, vec4(newWaterHeight, 0.0, 0.0, 0.0));
}
"""
shader_sources['evaporate'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float evaporationConstant;
uniform float dt;
layout(r16f) uniform readonly image2D heightIn;
layout(r16f) uniform writeonly image2D heightOut;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 newWaterHeight = imageLoad(heightIn, coord) * (1.0 - evaporationConstant * dt);
  imageStore(heightOut, coord, newWaterHeight);
}
"""
shader_sources['update_main_data'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

layout(r16f) uniform readonly image2D heightNew;
layout(r16f) uniform writeonly image2D heightBase;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  imageStore(heightBase, coord, imageLoad(heightNew, coord));
}
"""
shader_sources['calculate_water_normals'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

layout(r16f) uniform readonly image2D terrainHeight;
layout(r16f) uniform readonly image2D waterHeight;
layout(rgba16f) uniform writeonly image2D normals;

float totalHeight(ivec2 uv) {
  return imageLoad(terrainHeight, uv).x + imageLoad(waterHeight, uv).x;
}

vec3 sobel(ivec2 uv) {
  // 0/2  1/2  2/2
  // 0/1       2/1
  // 0/0  1/0  2/0
  float h00 = totalHeight(uv + ivec2(-1, -1));
  float h01 = totalHeight(uv + ivec2(-1,  0));
  float h02 = totalHeight(uv + ivec2(-1,  1));
  float h10 = totalHeight(uv + ivec2( 0, -1));
  float h12 = totalHeight(uv + ivec2( 0,  1));
  float h20 = totalHeight(uv + ivec2( 1, -1));
  float h21 = totalHeight(uv + ivec2( 1,  0));
  float h22 = totalHeight(uv + ivec2( 1,  1));

  float x = atan(( h00 + 2*h01 + h02 - h20 - 2*h21 - h22));
  float y = atan(( h00 + 2*h10 + h20 - h02 - 2*h12 - h22));
  float z = 1.0 - x * x - y * y;
  vec3 normal = vec3((x + 1.0) / 2.0, (y + 1.0) / 2.0, (z + 1.0) / 2.0);
  return normal;
}

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  int gridSize = imageSize(waterHeight).x;
  vec4 normal = vec4(sobel(coord), 1.0);
  imageStore(normals, coord, normal);
}
"""


hyper_model_params = dict(
    add_water               = [],
    calculate_outflux       = ['boundary_condition'],
    apply_crossflux         = ['boundary_condition'],
    evaporate               = [],
    update_main_data        = [],
    calculate_water_normals = [], # FIXME: Boundary condition
)
default_hyper_model = dict(
    boundary_condition=BoundaryConditions.CLOSED, # Open outflow, wall, tiling
)


model_params = dict(
    add_water               = ['dt'],
    calculate_outflux       = ['dt', 'pipeCoefficient', 'cellDistance'],
    apply_crossflux         = ['dt', 'cellDistance'],
    evaporate               = ['dt', 'evaporationConstant'],
    update_main_data        = [],
    calculate_water_normals = [],
)
water_flow_model = (
    [
        ('terrain_height', 1),
        ('water_height', 1),
        ('water_influx', 1),
        ('water_height_after_influx', 1),
        ('water_crossflux', 4),
        ('water_height_after_crossflux', 1),
        ('water_height_after_evaporation', 1),
        ('water_normal_map', 4),
    ],
    {
        'add_water': dict(
            heightIn='water_height',
            influx='water_influx',
            heightOut='water_height_after_influx',
        ),
        'calculate_outflux': dict(
            terrainHeight='terrain_height',
            waterHeight='water_height_after_influx',
            waterCrossflux='water_crossflux',
        ),
        'apply_crossflux': dict(
            heightIn='water_height_after_influx',
            waterCrossflux='water_crossflux',
            heightOut='water_height_after_crossflux',
        ),
        'evaporate': dict(
            heightIn='water_height_after_crossflux',
            heightOut='water_height_after_evaporation',
        ),
        'update_main_data': dict(
            heightNew='water_height_after_evaporation',
            heightBase='water_height',
        ),
        'calculate_water_normals': dict(
            terrainHeight='terrain_height',
            waterHeight='water_height',
            normals='water_normal_map',
        ),
    },
)


class Simulation:
    def __init__(
            self,
            model=water_flow_model,
            hyper_model=default_hyper_model,
            resolution=256,
            dump_shaders=False,
            cell_distance=1.0,
            pipe_coefficient=98.1,
            evaporation_constant=0.01,
            terrain_height=None,
            water_height=None,
            water_influx=None,
    ):
        self.resolution = resolution
        self.f_res = float(resolution)
        self.hyper_model = hyper_model
        self.images = {}
        self.textures = {}
        self.compute_nodes = {}

        # Set up the model
        self.dump_shaders = dump_shaders
        self.setup_model(model)

        # Apply model parameters
        self.cell_distance = cell_distance
        self.pipe_coefficient = pipe_coefficient
        self.evaporation_constant = evaporation_constant

    def make_map(self, name, channels=1):
        assert channels in [1, 4]
        image = PfmFile()
        image.clear(
            x_size=self.resolution,
            y_size=self.resolution,
            num_channels=channels,
        )
        if channels == 1:
            image.fill(0.0)
        else:
            image.fill((0.0, 0.0, 0.0, 0.0))
        self.images[name] = image

        texture = Texture(name)
        self.textures[name] = texture
        self.load_image(name)
        return texture, image

    def load_image(self, name):
        image = self.images[name]
        texture = self.textures[name]
        texture.load(image)
        if image.num_channels == 1:
            texture.set_format(Texture.F_r16)
        else:
            texture.set_format(Texture.F_rgba16)
        texture.wrap_u = Texture.WM_clamp
        texture.wrap_v = Texture.WM_clamp
    
        return texture, image
        
    def setup_model(self, model):
        data, process = model
        for (name, channels) in data:
            self.textures[name], self.images[name] = self.make_map(name, channels=channels)
        for cull_bin_idx, (name, shader_params) in enumerate(process.items()):
            cull_bin_sort = -len(process) + cull_bin_idx
            self.compute_nodes[name] = self.add_compute_node(name, cull_bin_sort, shader_params)

    def add_compute_node(self, name, cull_bin_sort, shader_params):
        shader_template = Template(shader_sources[name])
        render_params = {}
        for hyper_parameter in hyper_model_params[name]:
            render_params[hyper_parameter] = self.hyper_model[hyper_parameter]
            if hyper_parameter == 'boundary_condition':
                render_params['BoundaryConditions'] = BoundaryConditions
        shader_source = shader_template.render(**render_params)
        if self.dump_shaders:
            print(f"----- {name} -----")
            for line_num, line in enumerate(shader_source.split('\n')):
                print(f"{line_num + 1 : 04d} {line}")
        compute_shader = Shader.make_compute(
            Shader.SL_GLSL,
            shader_source,
        )
        compute_shader.set_filename(Shader.ST_none, name)
        workgroups = (self.resolution // 16, self.resolution // 16, 1)
        compute_node = ComputeNode("compute")
        compute_node.add_dispatch(*workgroups)
        compute_np = NodePath(compute_node)
        compute_np.set_shader(compute_shader)
        for glsl_name, py_name in shader_params.items():
            compute_np.set_shader_input(glsl_name, self.textures[py_name])
        compute_np.setBin("fixed", cull_bin_sort)
        return compute_np

    def attach_compute_nodes(self, root_np):
        for compute_np in self.compute_nodes.values():
            compute_np.reparent_to(root_np)

    def print_mem_usage(self):
        mem_use = sum([t.estimate_texture_memory() for t in self.textures.values()])
        print(f"Memory use: {mem_use} bytes")

    def set_shader_param(self, parameter, value):
        for shader_name, shader_params in model_params.items():
            if parameter in shader_params:
                self.compute_nodes[shader_name].set_shader_input(parameter, value)

    @property
    def dt(self):
        self._dt = value

    @dt.setter
    def dt(self, value):
        self._dt = value
        self.set_shader_param('dt', value)

    @property
    def cell_distance(self):
        return self._cell_distance

    @cell_distance.setter
    def cell_distance(self, value):
        self._cell_distance = value
        self.set_shader_param('cellDistance', value)

    @property
    def pipe_coefficient(self):
        return self._pipe_coefficient

    @pipe_coefficient.setter
    def pipe_coefficient(self, value):
        self._pipe_coefficient = value
        self.set_shader_param('pipeCoefficient', value)

    @property
    def evaporation_constant(self):
        return self._evaporation_constant

    @evaporation_constant.setter
    def evaporation_constant(self, value):
        self._evaporation_constant = value
        self.set_shader_param('evaporationConstant', value)
