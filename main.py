import random

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from panda3d.core import load_prc_file_data
from panda3d.core import PStatClient
from panda3d.core import KeyboardButton
from panda3d.core import TextNode
from panda3d.core import CardMaker
from panda3d.core import NodePath
from panda3d.core import Vec3
from panda3d.core import LColor

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText

from simulator import BoundaryConditions
from simulator import Simulation
from visuals import make_terrain
import make_heightmaps
from model import cutting_edge_model as model
#from model import water_flow_model as model


hyper_params, model_params, _, _ = model


defaults = "Defaults:\n"
defaults += '\n'.join([
    f"{name}: {value}"
    for name, value in hyper_params.items()
])
defaults += "\n"
defaults += '\n'.join([
    f"{name}: {value}"
    for name, value in model_params.items()
])
parser = ArgumentParser(
    description="Hydraulic erosion simulation for Panda3D.",
    formatter_class=RawDescriptionHelpFormatter,
    epilog=defaults
)
parser.add_argument(
    '-R',
    '--resolution',
    type=int,
    help='Side length of the simulation in cells.',
)
parser.add_argument(
    '-W',
    '--workgroup',
    type=int,
    nargs=2,
    help='Workgroup size.',
)
parser.add_argument(
    '-P',
    '--precision',
    type=int,
    choices=[16, 32],
    help='Data precision used (half float or float).',
)
parser.add_argument(
    '-B',
    '--boundary-condition',
    choices=['open', 'closed', 'wrap'],
    help="Type of the boundary at the map's edge.",
)
parser.add_argument(
    '-t',
    '--timestep',
    type=float,
    help='Time step. Omit to use wall time.',
)
parser.add_argument(
    '-p',
    '--pipe-coefficient',
    type=float,
    help='Pipe coefficient; Gravity * crossarea of the pipe / its length.',
)
parser.add_argument(
    '-c',
    '--cell-distance',
    type=float,
    help='Side length of each cell.',
)
parser.add_argument(
    '-s',
    '--sediment-capacity',
    type=float,
    help='How much sediment each unit of water can carry.',
)
parser.add_argument(
    '-e',
    '--erosion-coefficient',
    type=float,
    help='',
)
parser.add_argument(
    '-d',
    '--deposition-coefficient',
    type=float,
    help='',
)
parser.add_argument(
    '-l',
    '--lower-tilt-bound',
    type=float,
    help='',
)
parser.add_argument(
    '-v',
    '--evaporation-constant',
    type=float,
    help='Fraction of of total water in a cell that is evaporated in a second.',
)
parser.add_argument(
    '-M',
    '--memory',
    action='store_true',
    help='Print memory use.',
)
parser.add_argument(
    '-S',
    '--dump-shaders',
    action='store_true',
    help='Print shader source code.',
)
args = parser.parse_args()


## Hyper parameters
if args.workgroup is not None:
    hyper_params['workgroup'] = args.workgroup
if args.precision is not None:
    hyper_params['precision'] = args.precision
if args.resolution is not None:
    hyper_params['resolution'] = args.resolution
if args.boundary_condition is not None:
    hyper_params['boundary_condition'] = {
        'open': BoundaryConditions.OPEN,
        'closed': BoundaryConditions.CLOSED,
        'wrap': BoundaryConditions.WRAPPING,
    }[args.boundary_condition]
# Model parameters
if args.timestep is not None:
    model_params['dt'] = args.timestep
else:  # We need a scalar value to set up the shaders; We'll replace it with realtime values at runtime
    model_params['dt'] = 0.0
if args.pipe_coefficient is not None:
    model_params['pipe_coefficient'] = args.pipe_coefficient
if args.cell_distance is not None:
    model_params['cell_distance'] = args.cell_distance
if args.sediment_capacity is not None:
    model_params['sediment_capacity'] = args.sediment_capacity
if args.erosion_coefficient is not None:
    model_params['erosion_coefficient'] = args.erosion_coefficient
if args.deposition_coefficient is not None:
    model_params['deposition_coefficient'] = args.deposition_coefficient
if args.lower_tilt_bound is not None:
    model_params['lower_tilt_bound'] = args.lower_tilt_bound
if args.evaporation_constant is not None:
    model_params['evaporation_constant'] = args.evaporation_constant


# Create the simulator
if args.dump_shaders:
    print("dump")
    simulator = Simulation(model, dump_shaders=True)
else:
    print("No dump")
    simulator = Simulation(model)
if args.memory:
    simulator.print_mem_usage()


# "Complex" terrain
edge_length = simulator.resolution
base_frequency = 256.0 / edge_length
min_height = 0.5
max_height = base_frequency + min_height
make_heightmaps.perlin_terrain(simulator.images['terrain_height'], min_height, max_height, base_freq=base_frequency)

#min_height, max_height = 0.0, 1.0
#make_heightmaps.perlin(simulator.images['terrain_height'])
#make_heightmaps.funnel(simulator.images['terrain_height'])
#make_heightmaps.sine_hills(simulator.images['terrain_height'], phases=1)
#make_heightmaps.block(simulator.images['terrain_height'])
#make_heightmaps.half(simulator.images['terrain_height'])
#make_heightmaps.half(simulator.images['water_height'])

simulator.load_image('terrain_height')
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
terrain = make_terrain(simulator, resolution=resolution, min_height=min_height, max_height=max_height)
terrain.reparent_to(base.render)
simulator.attach_compute_nodes(terrain)


def spring(image):
    influx_mass = 0.5
    influx_randomness = influx_mass * 0.25
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
        if args.timestep is None:
            base.task_mgr.add(self.set_simulator_dt, sort=-5)
        # Influx
        self.fountain = False
        self.rain = 0
        base.task_mgr.add(self.update_influx, sort=-5)
        base.accept("f", self.toggle_fountain)
        base.accept("r", self.toggle_rain)
        # Camera
        base.cam.node().get_lens().near = 0.001
        base.camera.set_pos(0, 0, 0)
        base.camera.set_p(-30)
        base.cam.set_pos(0, -5, 0)
        base.task_mgr.add(self.rotate_camera)
        # GUI
        base.set_frame_rate_meter(True)
        self.setup_help_text()
        self.setup_data_viewer()
        base.accept("v", self.toggle_data_map)

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
        self.help_text_map_viewer = OnscreenText(
            text=f"V to change viewed data map (currently - / {len(self.simulator.model[2])})",
            parent=base.a2dTopLeft,
            align=TextNode.ALeft,
            pos=(0.01, -0.20),
            scale=0.05,
        )

    def setup_data_viewer(self):
        self.viewed_map = None
        self.viewer = NodePath(CardMaker('data viewer').generate())
        self.viewer.reparent_to(base.a2dBottomLeft)
        self.viewer.hide()
        self.help_text_peeker_value = OnscreenText(
            text=f"",
            font=base.loader.load_font('CourierPrimeCode.ttf'),
            align=TextNode.ALeft,
            parent=self.viewer,
            pos=(0.0, 1.02),
            scale=0.05,
        )
        base.task_mgr.add(self.peek_data_map)
        
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

    def toggle_data_map(self):
        num_maps = len(self.simulator.model[2])
        # Start / increase / reset counter
        if self.viewed_map is None:
            self.viewed_map = 0
            self.viewer.show()
        else:
            self.viewed_map += 1
            if self.viewed_map == num_maps:
                self.viewed_map = None
                self.viewer.hide()
        # Update viewer and help text
        if self.viewed_map is None:
            self.help_text_map_viewer['text'] = f"V to change viewed data map (currently - / {num_maps})"
            self.peeker_map = None
        else:
            map_name = self.simulator.model[2][self.viewed_map][0]
            data_map = self.simulator.textures[map_name]
            self.viewer.set_texture(data_map)
            self.help_text_map_viewer['text'] = f"V to change viewed data map (currently {self.viewed_map + 1} / {num_maps}, '{map_name}')"
            self.peeker_map = data_map

    def peek_data_map(self, task):
        if self.viewed_map is not None and base.mouseWatcherNode.hasMouse():
            mouse_coord = base.mouseWatcherNode.get_mouse()
            viewer_coord = self.viewer.get_relative_point(base.cam2d, Vec3(mouse_coord.x, 0, mouse_coord.y))
            if 0.0 <= viewer_coord.x <= 1.0 and 0.0 <= viewer_coord.z <= 1.0:
                color = LColor()
                base.graphicsEngine.extract_texture_data(
                    self.peeker_map,
                    base.win.get_gsg(),
                )
                self.peeker = self.peeker_map.peek()
                self.peeker.lookup(color, viewer_coord.x, viewer_coord.z)
                num_channels = self.peeker_map.num_components
                if num_channels == 1:
                    data_text = f"{color.x: 2.4f}"
                elif num_channels == 2:
                    data_text = f"{color.x: 2.4f}  {color.y: 2.4f}"
                elif num_channels == 4:
                    data_text = f"{color.x: 2.4f}  {color.y: 2.4f}  {color.z: 2.4f}  {color.w: 2.4f}"
                self.help_text_peeker_value['text'] = data_text
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
