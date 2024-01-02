sobel_normals = """
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

  float x = sin(atan((h00 + 2*h01 + h02 - h20 - 2*h21 - h22)));
  float y = sin(atan((h00 + 2*h10 + h20 - h02 - 2*h12 - h22)));
  float z = sqrt(1.0 - x * x - y * y);
  vec3 normal = vec3(x, y, z);
  normal *= 0.5;
  normal += 0.5;
  return normal;
}

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec4 normal = vec4(sobel(coord), 1.0);
  imageStore(normals, coord, normal);
}
"""


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
layout(rg16f) uniform writeonly image2D waterVelocity;
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
  float inflowLeft   = imageLoad(waterCrossflux, wrapCoord(coord + ivec2(-1,  0))).g;
  float inflowRight  = imageLoad(waterCrossflux, wrapCoord(coord + ivec2( 1,  0))).r;
  float inflowBottom = imageLoad(waterCrossflux, wrapCoord(coord + ivec2( 0, -1))).a;
  float inflowTop    = imageLoad(waterCrossflux, wrapCoord(coord + ivec2( 0,  1))).b;
  float totalInflow = inflowLeft + inflowRight + inflowBottom + inflowTop;
  vec4 outflows = imageLoad(waterCrossflux, coord);

  // Update water height
  float totalOutflow = outflows.r + outflows.g + outflows.b + outflows.a;
  float deltaVolume = (totalInflow - totalOutflow) * dt;
  float oldWaterHeight = imageLoad(heightIn, coord).x;
  float newWaterHeight = oldWaterHeight + deltaVolume / (cellDistance * cellDistance);
  imageStore(heightOut, coord, vec4(newWaterHeight, 0.0, 0.0, 0.0));

  // Velocity
  vec2 cellWaterThroughputPerSecond = vec2(
    inflowLeft + outflows.g - inflowRight - outflows.r,
    inflowBottom + outflows.a - inflowTop - outflows.b
  ) * 0.5;
  float averageWaterHeight = (oldWaterHeight + newWaterHeight) / 2.0;
  vec2 velocity = cellWaterThroughputPerSecond / (averageWaterHeight * cellDistance);
  imageStore(waterVelocity, coord, vec4(velocity, 0.0, 0.0));
}
"""


shader_sources['erode_deposit'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;
uniform float sedimentCapacity;
uniform float erosionCoefficient;
uniform float depositionCoefficient;
uniform float lowerTiltBound;

layout(r16f) uniform readonly image2D terrainHeightIn;
layout(r16f) uniform writeonly image2D terrainHeightOut;
layout(rgba16f) uniform readonly image2D normals;
layout(rg16f) uniform readonly image2D waterVelocity;
layout(r16f) uniform readonly image2D sedimentIn;
layout(r16f) uniform writeonly image2D sedimentOut;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  float terrain = imageLoad(terrainHeightIn, coord).x;
  float velocity = imageLoad(waterVelocity, coord).x;
  float sediment = imageLoad(sedimentIn, coord).x;
  float tilt = acos(dot(vec3(0.0, 0.0, 1.0), imageLoad(normals, coord).xyz * 2.0 - 1.0));

  float sedimentTransportCapacity = max(lowerTiltBound, sedimentCapacity * sin(tilt) * length(velocity));
  float deltaCapacity = sediment - sedimentTransportCapacity;
  float massEroded = max(0.0, deltaCapacity * - 1.0) * erosionCoefficient;
  float massDeposited = max(0.0, deltaCapacity) * depositionCoefficient;
  float deltaSuspendedMass = (massEroded - massDeposited) * dt;

  imageStore(sedimentOut, coord, vec4(sediment + deltaSuspendedMass));
  imageStore(terrainHeightOut, coord, vec4(terrain - deltaSuspendedMass));
}
"""


shader_sources['transport_solute'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

uniform float dt;

uniform sampler2D soluteIn;
layout(rg16f) uniform readonly image2D velocityMap;
layout(r16f) uniform writeonly image2D soluteOut;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  vec2 velocity = imageLoad(velocityMap, coord).xy;
  vec2 fcoord = vec2(coord);
  float newSoluteAmount = texture(soluteIn, fcoord - velocity * dt).x;
  imageStore(soluteOut, coord, vec4(newSoluteAmount, 0.0, 0.0, 0.0));
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


shader_sources['calculate_terrain_normals'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

layout(r16f) uniform readonly image2D terrainHeight;
layout(rgba16f) uniform writeonly image2D normals;

float totalHeight(ivec2 uv) {
  return imageLoad(terrainHeight, uv).x;
}
"""+sobel_normals


shader_sources['calculate_water_normals'] = """
#version 430

layout (local_size_x=16, local_size_y=16) in;

layout(r16f) uniform readonly image2D terrainHeight;
layout(r16f) uniform readonly image2D waterHeight;
layout(rgba16f) uniform writeonly image2D normals;

float totalHeight(ivec2 uv) {
  return imageLoad(terrainHeight, uv).x + imageLoad(waterHeight, uv).x;
}
"""+sobel_normals
