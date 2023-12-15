from argparse import ArgumentParser

from panda3d.core import load_prc_file_data
from panda3d.core import PStatClient

from direct.showbase.ShowBase import ShowBase

from simulator import BoundaryConditions
from simulator import Simulation
from visuals import make_terrain
import make_heightmaps


parser = ArgumentParser(
    description="Hydraulics simulation for Panda3D.",
    epilog="",
)
parser.add_argument(
    '-r',
    '--resolution',
    type=int,
    help='Side length of the simulation in cells.',
)
parser.add_argument(
    '-e',
    '--evaporation',
    type=float,
    help='Fraction of of total water in a cell that is evaporated in a second.',
)
parser.add_argument(
    '-p',
    '--pipe',
    type=float,
    help='Pipe coefficient; Gravity * crossarea of the pipe / its length.',
)
parser.add_argument(
    '-t',
    '--timestep',
    type=float,
    help='Time step.',
)
parser.add_argument(
    '-m',
    '--memory',
    action='store_true',
    help='Print memory use.',
)
parser.add_argument(
    '-s',
    '--shaders',
    action='store_true',
    help='Print shader source code.',
)
args = parser.parse_args()

# Create the simulation
simulator_kwargs = {}
if args.resolution is not None:
    simulator_kwargs['resolution'] = args.resolution
if args.evaporation is not None:
    simulator_kwargs['evaporation_constant'] = args.evaporation
if args.pipe is not None:
    simulator_kwargs['pipe_coefficient'] = args.pipe
if args.shaders:
    simulator_kwargs['dump_shaders'] = True
simulator = Simulation(
    hyper_model=dict(boundary_condition=BoundaryConditions.CLOSED),  # OPEN, CLOSED, or WRAPPING
    **simulator_kwargs,
)
if args.memory:
    simulator.print_mem_usage()


make_heightmaps.perlin(simulator.images['terrain_height'])
#make_heightmaps.funnel(simulator.images['terrain_height'])
#make_heightmaps.sine_hills(simulator.images['terrain_height'], phases=2)
simulator.load_image('terrain_height')


load_prc_file_data('', 'gl-version 3 2')
ShowBase()
base.disable_mouse()
base.accept('escape', base.task_mgr.stop)
base.pstats = True
PStatClient.connect()


# Set timestep on the simulation / enable wall time steps
def set_simulator_dt(task):
    simulator.dt = globalClock.dt
    return task.cont
if args.timestep is not None:
    print(f"Timestep: {args.timestep}")
    simulator.dt = args.timestep
else:
    print(f"Timestep: realtime")
    base.task_mgr.add(set_simulator_dt, sort=-5)


if simulator.resolution > 256:
    resolution = 256
else:
    resolution = simulator.resolution
terrain = make_terrain(simulator, resolution=resolution)
terrain.reparent_to(base.render)
simulator.attach_compute_nodes(terrain)
#simulator.attach_compute_nodes(base.render)


# And we want to stop the influx of water, and begin evaporation.
global influx
influx = False
def toggle_influx():
    global influx
    influx = not influx
    influx_mass = 300.0
    water_influx = simulator.images['water_influx']
    resolution = simulator.resolution
    if influx:
        water_influx.set_point1(simulator.resolution // 2, simulator.resolution // 2, influx_mass)
    else:
        water_influx.fill(0.0)
    simulator.load_image('water_influx')
base.accept("space", toggle_influx)


# Camera needs some love, too
#base.cam.set_pos(0, -1, 4)
base.cam.set_pos(2, -2, 2)
base.cam.look_at(0, 0, 0.25)
base.set_frame_rate_meter(True)
base.run()
