from argparse import ArgumentParser

from panda3d.core import load_prc_file_data

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
    help='Side length of the simulation in cells.',
)
parser.add_argument(
    '-e',
    '--evaporation',
    help='Fraction of of total water in a cell that is evaporated in a second.',
)
args = parser.parse_args()

simulator_kwargs = {}
if args.resolution is not None:
    simulator_kwargs['resolution'] = int(args.resolution)
if args.evaporation is not None:
    simulator_kwargs['evaporation_constant'] = float(args.evaporation)


simulator = Simulation(
    hyper_model=dict(boundary_condition=BoundaryConditions.CLOSED),  # OPEN, CLOSED, or WRAPPING
    **simulator_kwargs,
)
simulator.dt = 1.0 / 60.0
simulator.print_mem_usage()


make_heightmaps.perlin(simulator.images['terrain_height'])
#make_heightmaps.funnel(simulator.images['terrain_height'])
#make_heightmaps.sine_hills(simulator.images['terrain_height'], phases=2)
simulator.load_image('terrain_height')


load_prc_file_data('', 'gl-version 3 2')
ShowBase()
base.disable_mouse()
base.accept('escape', base.task_mgr.stop)

terrain = make_terrain(simulator)
terrain.reparent_to(base.render)
simulator.attach_compute_nodes(terrain)


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
