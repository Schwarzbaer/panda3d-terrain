import random

from argparse import ArgumentParser

from panda3d.core import load_prc_file_data
from panda3d.core import PStatClient
from panda3d.core import KeyboardButton
from panda3d.core import TextNode

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText

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
parser.add_argument(
    '-b',
    '--boundary',
    choices=['open', 'closed', 'wrap'],
    default='closed',
    help="Select the map's boundary type.",
)
args = parser.parse_args()

# Create the simulation
model_kwargs = {}
if args.resolution is not None:
    model_kwargs['resolution'] = args.resolution
if args.evaporation is not None:
    model_kwargs['evaporation_constant'] = args.evaporation
if args.pipe is not None:
    model_kwargs['pipe_coefficient'] = args.pipe
if args.shaders:
    model_kwargs['dump_shaders'] = True
hyper_model_kwargs = dict(boundary_condition={
    'open': BoundaryConditions.OPEN,
    'closed': BoundaryConditions.CLOSED,
    'wrap': BoundaryConditions.WRAPPING,
}[args.boundary])
simulator = Simulation(
    hyper_model=hyper_model_kwargs,
    **model_kwargs,
)
if args.memory:
    simulator.print_mem_usage()


make_heightmaps.perlin(simulator.images['terrain_height'])
#make_heightmaps.funnel(simulator.images['terrain_height'])
#make_heightmaps.sine_hills(simulator.images['terrain_height'], phases=1)
#make_heightmaps.block(simulator.images['terrain_height'])
#make_heightmaps.half(simulator.images['terrain_height'])
simulator.load_image('terrain_height')
#make_heightmaps.half(simulator.images['water_height'])
#simulator.load_image('water_height')


load_prc_file_data('', 'gl-version 3 2')
load_prc_file_data('', 'pstats-gpu-timing #t')
ShowBase()
base.disable_mouse()
base.accept('escape', base.task_mgr.stop)
base.pstats = True
PStatClient.connect()


if simulator.resolution > 256:
    resolution = 256
else:
    resolution = simulator.resolution
terrain = make_terrain(simulator, resolution=resolution)
terrain.reparent_to(base.render)
simulator.attach_compute_nodes(terrain)
#simulator.attach_compute_nodes(base.render)


def spring(image):
    influx_mass = 3.0
    influx_randomness = 1.0
    influx_area = 5
    resolution = simulator.resolution
    offset = simulator.resolution //2 - influx_area // 2
    for x in range(offset, offset + influx_area + 1):
        for y in range(offset, offset + influx_area + 1):
            mass = influx_mass + (random.random() - 0.5) * influx_randomness
            image.set_point1(x, y, mass)


def equal_rain(image):
    image.fill(0.01)


class Interface:
    def __init__(self, simulator):
        self.simulator = simulator
        # Timestep
        if args.timestep is not None:
            print(f"Timestep: {args.timestep}")
            self.simulator.dt = args.timestep
        else:
            print(f"Timestep: realtime")
            base.task_mgr.add(self.set_simulator_dt, sort=-5)
        # Influx
        self.influx = False
        base.task_mgr.add(self.update_influx, sort=-5)
        base.accept("space", self.toggle_influx)
        # Camera
        base.cam.set_pos(0, -2, 2)
        base.cam.look_at(0, 0, 0.25)
        base.task_mgr.add(self.rotate_camera)
        # Frame rate meter
        base.set_frame_rate_meter(True)

    # Set timestep on the simulation / enable wall time steps
    def set_simulator_dt(self, task):
        self.simulator.dt = globalClock.dt
        return task.cont

    def toggle_influx(self):
        self.influx = not self.influx
        if not self.influx:
            image = self.simulator.images['water_influx']
            image.fill(0.0)
            self.simulator.load_image('water_influx')

    def update_influx(self, task):
        if self.influx:
            image = self.simulator.images['water_influx']
            spring(image)
            #equal_rain(image)
            self.simulator.load_image('water_influx')
        return task.cont

    def rotate_camera(self, task):
        if base.mouseWatcherNode.has_mouse():
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('a')):
                base.camera.set_h(base.camera.get_h() - 60.0 * globalClock.dt)
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('d')):
                base.camera.set_h(base.camera.get_h() + 60.0 * globalClock.dt)
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('w')):
                base.camera.set_p(base.camera.get_p() - 30.0 * globalClock.dt)
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('s')):
                base.camera.set_p(base.camera.get_p() + 30.0 * globalClock.dt)
        return task.cont


Interface(simulator)
help_text_water = OnscreenText(
    text='Space to toggle water influx.',
    parent=base.a2dTopLeft,
    align=TextNode.ALeft,
    pos=(0.01, -0.05),
    scale=0.05,
)
help_text_water = OnscreenText(
    text='WASD to rotate the terrain.',
    parent=base.a2dTopLeft,
    align=TextNode.ALeft,
    pos=(0.01, -0.1),
    scale=0.05,
)
base.run()
