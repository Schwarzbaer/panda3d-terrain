import enum

from jinja2 import Template

from panda3d.core import NodePath
from panda3d.core import PfmFile
from panda3d.core import Texture
from panda3d.core import Shader
from panda3d.core import ComputeNode

from model import BoundaryConditions
from model import cutting_edge_model
from shaders import shader_sources


def glslify(name):
    constituents = name.split('_')
    glsl_variant = [str.capitalize(c) for c in constituents]
    glsl_variant[0] = constituents[0]
    return ''.join(glsl_variant)


class Simulation:
    def __init__(self, model=cutting_edge_model, dump_shaders=False):
        self.model = model
        self.dump_shaders = dump_shaders

        self.images = {}
        self.textures = {}
        self.compute_nodes = {}

        self.setup_maps(model)
        self.setup_processes(model)
        self.setup_model_parameters(model)

    def setup_maps(self, model):
        hyper_params, _, maps, _ = model
        resolution = hyper_params['resolution']
        for name, channels in maps:
            assert channels in [1, 2, 4]
            image = PfmFile()
            image.clear(
                x_size=resolution,
                y_size=resolution,
                num_channels=channels,
            )
            if channels == 1:
                image.fill(0.0)
            elif channels == 2:
                image.fill((0.0, 0.0))
            else:
                image.fill((0.0, 0.0, 0.0, 0.0))
            self.images[name] = image
    
            texture = Texture(name)
            self.textures[name] = texture
            self.load_image(name)

    def print_mem_usage(self):
        mem_use = sum([t.estimate_texture_memory() for t in self.textures.values()])
        print(f"Memory use: {mem_use} bytes")

    def load_image(self, name):
        image = self.images[name]
        texture = self.textures[name]
        texture.load(image)
        if image.num_channels == 1:
            texture.set_format(Texture.F_r16)
        elif image.num_channels == 2:
            texture.set_format(Texture.F_rg16)
        else:
            texture.set_format(Texture.F_rgba16)
        texture.wrap_u = Texture.WM_clamp
        texture.wrap_v = Texture.WM_clamp
    
        return texture, image
        
    def setup_processes(self, model):
        hyper_params, _, _, processes = model
        for cull_bin_idx, (step_name, process) in enumerate(processes.items()):
            shader_source_name, hyper_params_used, _, data_mapping = process
            cull_bin_sort = -len(processes) + cull_bin_idx
            shader_template = Template(shader_sources[shader_source_name])
            render_params = {}
            for hyper_param_name in hyper_params_used:
                render_params[hyper_param_name] = hyper_params[hyper_param_name]
                if hyper_param_name == 'boundary_condition':
                    render_params['BoundaryConditions'] = BoundaryConditions
            shader_source = shader_template.render(**render_params)
            if self.dump_shaders:
                print(f"----- {step_name} -----")
                for line_num, line in enumerate(shader_source.split('\n')):
                    print(f"{line_num + 1 : 04d} {line}")
            compute_shader = Shader.make_compute(
                Shader.SL_GLSL,
                shader_source,
            )
            compute_shader.set_filename(Shader.ST_none, step_name)
            resolution = hyper_params['resolution']
            workgroups = (resolution // 16, resolution // 16, 1)
            compute_node = ComputeNode(step_name)
            compute_node.add_dispatch(*workgroups)
            compute_np = NodePath(compute_node)
            compute_np.set_shader(compute_shader)
            for glsl_name, py_name in data_mapping.items():
                compute_np.set_shader_input(glsl_name, self.textures[py_name])
            compute_np.setBin("fixed", cull_bin_sort)
            self.compute_nodes[step_name] = compute_np

    def attach_compute_nodes(self, root_np):
        for compute_np in self.compute_nodes.values():
            compute_np.reparent_to(root_np)

    def setup_model_parameters(self, model):
        hyper_params, model_params, _, processes = model
        def getter_func(name):
            def inner(self):
                return getattr(self, f'_{name}')
            return inner
        def setter_func(name):
            def inner(self, value):
                setattr(self, f'_{name}', value)
                for step_name, process in processes.items():
                    _, _, model_params, _ = process
                    if name in model_params:
                        compute_node = self.compute_nodes[step_name]
                        glsl_name = glslify(name)
                        compute_node.set_shader_input(glsl_name, value)
            return inner
        for param_name, value in hyper_params.items():
            setattr(
                self.__class__,
                param_name,
                property(fget=getter_func(param_name)),
            )
            setattr(self, f'_{param_name}', value)
        for param_name, value in model_params.items():
            setattr(
                self.__class__,
                param_name,
                property(
                    fget=getter_func(param_name),
                    fset=setter_func(param_name),
                ),
            )
            setattr(self, param_name, value)

    def set_model_parameter(self, name, value):
        setattr(self, f'_{name}', value)
        _, _, _, processes = self.model
        for step_name, process in processes.items():
            _, _, model_params, _ = process
            if name in model_params:
                self.compute_nodes[step_name].set_shader_input(name, value)
