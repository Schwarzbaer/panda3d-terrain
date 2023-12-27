import enum


class BoundaryConditions(enum.Enum):
    OPEN = 1
    CLOSED = 2
    WRAPPING = 3


hyper_defaults = dict(
    resolution=256,
    boundary_condition=BoundaryConditions.OPEN,
)
model_defaults = dict(
    dt=None,
    pipe_coefficient=98.1,
    cell_distance=1.0,
    sediment_capacity=0.0005,
    erosion_coefficient=0.0005,
    deposition_coefficient=0.0005,
    lower_tilt_bound=0.0,
    evaporation_constant=0.05,
)
maps = [
    ('terrain_height', 1),
    ('water_height', 1),
    ('water_influx', 1),
    ('water_height_after_influx', 1),
    ('water_crossflux', 4),
    ('water_velocity', 2),
    ('water_height_after_crossflux', 1),
    ('terrain_height_after_erosion_deposition', 1),
    ('suspended_sediment', 1),
    ('suspended_sediment_after_transport', 1),
    ('water_height_after_evaporation', 1),
    ('terrain_normal_map', 4),
    ('water_normal_map', 4),
]
processes = dict(
    add_water=(                       # step name
        'add_water',                  # shader source
        [],                           # hyper parameters
        ['dt'],                       # model parameters,
        dict(
            heightIn='water_height',  # glsl input name='map_name'
            influx='water_influx',
            heightOut='water_height_after_influx',
        ),
    ),
    calculate_outflux=(
        'calculate_outflux',
        ['boundary_condition'],
        ['dt', 'pipe_coefficient', 'cell_distance'],
        dict(
            terrainHeight='terrain_height',
            waterHeight='water_height_after_influx',
            waterCrossflux='water_crossflux',
        ),
    ),
    apply_crossflux=(
        'apply_crossflux',
        ['boundary_condition'],
        ['dt', 'cell_distance'],
        dict(
            heightIn='water_height_after_influx',
            waterCrossflux='water_crossflux',
            waterVelocity='water_velocity',
            heightOut='water_height_after_crossflux',
        ),
    ),
    erode_deposit=(
        'erode_deposit',
        [],
        ['dt', 'sediment_capacity', 'erosion_coefficient', 'deposition_coefficient', 'lower_tilt_bound'],
        dict(
            terrainHeightIn='terrain_height',
            terrainHeightOut='terrain_height_after_erosion_deposition',
            normals='terrain_normal_map',
            waterVelocity='water_velocity',
            sedimentIn='suspended_sediment',
            sedimentOut='suspended_sediment_after_transport',
        ),
    ),
    evaporate=(
        'evaporate',
        [],
        ['dt', 'evaporation_constant'],
        dict(
            heightIn='water_height_after_crossflux',
            heightOut='water_height_after_evaporation',
        ),
    ),
    update_terrain_height=(
        'update_main_data',
        [],
        [],
        dict(
            heightNew='terrain_height_after_erosion_deposition',
            heightBase='terrain_height',
        ),
    ),
    update_water_height=(
        'update_main_data',
        [],
        [],
        dict(
            heightNew='water_height_after_evaporation',
            heightBase='water_height',
        ),
    ),
    update_sediment=(
        'update_main_data',
        [],
        [],
        dict(
            heightNew='suspended_sediment_after_transport',
            heightBase='suspended_sediment',
        ),
    ),
    calculate_terrain_normals=(
        'calculate_terrain_normals',
        ['boundary_condition'],
        [],
        dict(
            terrainHeight='terrain_height',
            normals='terrain_normal_map',
        ),
    ),
    calculate_water_normals=(
        'calculate_water_normals',
        ['boundary_condition'],
        [],
        dict(
            terrainHeight='terrain_height',
            waterHeight='water_height',
            normals='water_normal_map',
        ),
    ),
)

cutting_edge_model = (
    hyper_defaults,
    model_defaults,
    maps,
    processes,
)