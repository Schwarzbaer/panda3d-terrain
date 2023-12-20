import random

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

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
    epilog='\n'.join(
        [
            "Default values are:",
            "Resolution      : 256",
            "Boundary        : open",
            "Evaporation     : 0.05",
            "Pipe coefficient: 98.1 (1g, 10m^2 diameter)",
            "Timestep        : realtime",
        ],
    ),
    formatter_class=RawDescriptionHelpFormatter,
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
            image.set_point1(x, y, image.get_point1(x, y) + mass)


def rain(image, rain_level):
    influx_per_drop = 0.5
    base_drops = 10
    total_drops = base_drops * rain_level ** 2
    resolution = simulator.resolution
    for _ in range(total_drops):
        x = random.randrange(resolution)
        y = random.randrange(resolution)
        image.set_point1(x, y, image.get_point1(x, y) + influx_per_drop)


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
        self.fountain = False
        self.rain = 0
        base.task_mgr.add(self.update_influx, sort=-5)
        base.accept("f", self.toggle_fountain)
        base.accept("r", self.toggle_rain)
        # Camera
        base.cam.node().get_lens().near = 0.001
        base.camera.set_pos(0, 0, 0.25)
        base.camera.set_p(-30)
        base.cam.set_pos(0, -3, 0)
        base.task_mgr.add(self.rotate_camera)
        # GUI
        base.set_frame_rate_meter(True)
        self.setup_help_text()

    def setup_help_text(self):
        self.help_text_wasd = OnscreenText(
            text="WASD to rotate the terrain, QE to zoom.",
            parent=base.a2dTopLeft,
            align=TextNode.ALeft,
            pos=(0.01, -0.05),
            scale=0.05,
        )
        self.help_text_fountain = OnscreenText(
            text=f"F to toggle fountain (currently {self.fountain})",
            parent=base.a2dTopLeft,
            align=TextNode.ALeft,
            pos=(0.01, -0.10),
            scale=0.05,
        )
        self.help_text_rain = OnscreenText(
            text=f"R to change rain level (currently {self.rain} / 3)",
            parent=base.a2dTopLeft,
            align=TextNode.ALeft,
            pos=(0.01, -0.15),
            scale=0.05,
        )

    # Set timestep on the simulation / enable wall time steps
    def set_simulator_dt(self, task):
        self.simulator.dt = globalClock.dt
        return task.cont

    def toggle_fountain(self):
        self.fountain = not self.fountain
        self.help_text_fountain['text'] = f"F to toggle fountain (currently {self.fountain})"

    def toggle_rain(self):
        self.rain = (self.rain + 1) % 4
        self.help_text_rain['text'] = f"R to change rain level (currently {self.rain} / 3)"

    def update_influx(self, task):
        image = self.simulator.images['water_influx']
        image.fill(0.0)
        if self.fountain:
            spring(image)
        rain(image, self.rain)
        self.simulator.load_image('water_influx')
        return task.cont

    def rotate_camera(self, task):
        if base.mouseWatcherNode.has_mouse():
            h = base.camera.get_h()
            p = base.camera.get_p()
            d = -base.cam.get_y()
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('a')):
                h += 60.0 * globalClock.dt
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('d')):
                h -= 60.0 * globalClock.dt
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('w')):
                p = max(-90.0, p - 30.0 * globalClock.dt)
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('s')):
                p = min(0.0, p + 30.0 * globalClock.dt)
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('q')):
                d -= 1.05 * globalClock.dt
            if base.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key('e')):
                d += 1.05 * globalClock.dt
            base.camera.set_h(h)
            base.camera.set_p(p)
            base.cam.set_y(-d)
        return task.cont


Interface(simulator)
base.run()
