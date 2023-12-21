Terrain for Panda3D
===================

Just throwing some academic papers at the GPU to see what sticks. The
goal is to have a tool to generate and render beautiful natural
landscapes.

Current state: Alpha
* The lighting model is Lambertian diffusion with a static sun vector,
  the colors code-defined. It's basically the simplest hacked-together
  renderer imaginable.
* The environment model is trivial.
  * There is terrain generated from Perlin noise.
  * There is flowing water with rain and a fountain as a source.
* The simulation code is not optimized one bit.


Installation
------------

Requirements: `pip install panda3d Jinja2`

For this project itself... Clone the repo and addit to your virtualenv,
I guess? I haven't packaged anything yet.


Usage
-----

`python main.py`

There are several options, documented in `python main.py -h`.

The turbulence in frame time (`globalClock.dt` in Panda3D) introduces
turbulence in the simulation, so if you want a simulation that converges
on a mostly steady state, it is recommended to set the timestep to
approximate your framerate (assuming that you are running at a
resolution at which the simulation reaches realtime performance), e.g.
`python main.py -t 0.01666` for 60 frames per second.


Current Model
-------------

* Hypermodel
  * `boundary_condition`: Whether the edges of the simulation are...
    * `OPEN`: Water flows over the edge as if the terrain and water
      heights were `0.0` beyond it.
    * `CLOSED`: No water flows beyond the edge.
    * `WRAPPING`: Water flowing out of one edge of the map gets added on
      the opposite edge.
* Map Data
  * `terrain_height`: Height of the terrain
  * `water_height`: Current height of the water
  * `water_influx`: Water influx rate water column to simulate e.g.
    rain, or fountains. Values are the water column added per second.
* Intermediate Maps
  * `water_height_after_influx`
  * `water_crossflux`
    * r to -u
    * g to +u
    * b to -v
    * a to +v
  * `water_velocity`
    * r along u
    * g along v
  * `water_height_after_crossflux`
  * `water_height_after_evaporation`
  * `terrain_normal_map`: Surface normals in GLSL encoding.
  * `water_normal_map`: Surface normals in GLSL encoding.
* Scalar model parameters
  * `cell_distance`: Distance between centers of neighboring cells;
     Default `1.0`
  * `pipe_coefficient`: `gravity g * pipe cross area A / pipe length l`;
     Default `98.1` (Earth gravity `9.81 m/s**2`, pipe cross area
     `10 m**2`)
  * `evaporation_constant`: Percentage of water that would evaporate in
    one second if the amount of evaporation was constant. `0.0` for no
    evaporation. Default `0.05`
* Frame parameters
  * `dt`: time step. The amount of time for which the simulation will be
    advanced.
* Process
  * Influx of water: Add new water to the simulation.
    `water_height_after_influx = water_height + water_influx * dt`
  * Calculate outflux
    For each direction/channel: `water_outflux = current_outflux + height_difference * hight_difference * pipe_coefficient * dt`, lower border of `0.0`.
    If the amount of outflowing water is greater than that of water
    actually present in this cell, scale the outflux so that exactly
    zero water remains.
  * Crossflux of water
    `water_height_after_crossflux = water_height_after_influx - own water_crossflux + neighbors' water_crossflux to this cell`
  * Evaporate water: Remove a fraction of water from each cell.
    `water_height_after_evaporation = water_height_after_crossflux * (1 - evaporation_constant * dt)`
  * Update main data
    `waterHeight = waterHeightAfterEvaporation`


TODO
----

### Current hot topics

* Refactor simulation model and implementation. Adding a new step to the
  process should not require touching as many parts of the code.
* Aesthetics
  * Specular highlights
  * Fake SSS based on water depth
* Bug: Why is water leaking out of a CLOSED/WRAPPING simulation?


### Small nice-to-haves

* Command line parameters
  * cell_distance=1.0
  * sedimentCapacity=1.0
  * erosionCoefficient=1.0
  * sedimentationCoefficient=1.0
  * lowerTiltBound=0.0
* Simulation model
  * Velocity field
  * Erosion-deposition, lateral sediment transport
* Visualization Switcher
  * Water
    * Depth
    * Normals
    * Light model


### Icebox

* Aesthetics
  * Water
    * Side walls on closed/wrapped boundary
    * Waterfall on open boundary
  * Terrain
    * Side walls
* Hyper parameters
  * Initial data
    * Zero (current)
    * Generator shaders
    * Loaded images
  * Workgroup size
  * Data resolution (`r16f` vs. `r32f`)
* Simulation model
  * More papers
* Performance
  * Measure, and find the maximum simulation size / optimal workgroup
    size for realtime.
    * Requires: Make workgroup size a hyper parameter
  * Optimizations
    * Combine shaders into one
    * Instead of using array, use vector/matrix.
    * set up `shared` array for the work group that is 2 elements larger
      than the workgroup, making a one-element border around it. Preload
      global data into it, and do math. Write output back into global.
    * Red/Black mode
* Loading/saving simulations / dumping images
* Offline rendering, sidestepping the vsync limit
* Tiled simulation for offline rendering of super-large worlds
* Inflow map: R=inflow measured in height of water column, G=inflow by
  volume
* Outflow clamp map: Finer control over where and how boundary
  conditions occur.


Papers
------

* "Fast Hydraulic Erosion Simulation and Visualization on GPU": https://xing-mei.github.io/files/erosion.pdf
  * Overview
    * Influx of water
    * Calculating the pressure-based outflux from each cell to its neighbors
    * Update water height
    * Calculate water velocity
    * Erode material from terrain into water, or deposit it
    * Transport sediment through water flow
    * Evaporate water
  * Implementation
    * DONE: Increase water
    * DONE: Compute outflux flow; We treat neighboring cells as if they were connected by pipes.
    * DONE: Update water surface
    * DONE: Update velocity field
    * DONE: Erosion-deposition
    * Transport sediment
      new suspended sediment amount = s1 at position - uv * dt, interpolating the four nearest neighbors
    * DONE: Evaporate water
* "Fast Hydraulic and Thermal Erosion on GPU": http://diglib.eg.org/bitstream/handle/10.2312/EG2011.short.057-060/057-060.pdf?sequence=1
* "Hydraulic Erosion Simulation on the GPU for 3D terrains": https://www.diva-portal.org/smash/get/diva2:1646074/FULLTEXT01.pdf
* "Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion": https://inria.hal.science/hal-01262376/document
* "Procedural Modeling of the Great Barrier Reef": https://easychair.org/publications/preprint_download/qLmc
* "Desertscape Simulation": https://www.researchgate.net/profile/Axel-Paris/publication/335488341_Desertscape_Simulation/links/5db7f667a6fdcc2128e8d1d9/Desertscape-Simulation.pdf
  Adds a sand layer and a vegetation layer. Reproduces realistic dune formations.
* "Procedural Generation of Large-Scale Forests Using a Graph-Based Neutral Landscape Model": https://media.proquest.com/media/hms/PFT/1/Ijdn4?_s=ANyDa8athS4X9z8tNeVbdzsdiSQ%3D
  First skim: Segments a grip-based landscape into patches of wood species growth.
* "AutoBiomes: procedural generation of multi-biome landscapes": https://d-nb.info/121797170X/34
  Uses a climate simulation, then derives biomes from the results.
* "Efficient Animation of Water Flow on Irregular Terrains": http://www-cg.cis.iwate-u.ac.jp/lab/graphite06.pdf
* "Forming Terrains by Glacial Erosion": https://inria.hal.science/hal-04090644/file/Sigg23_Glacial_Erosion__author.pdf
* "Interactive Generation of Time-evolving, Snow-Covered Landscapes with Avalanches": https://inria.hal.science/hal-01736971/file/interactive-generation-time.pdf
* "Waterfall Simulation with Spray Cloud in different Environments": https://www.jstage.jst.go.jp/article/artsci/15/3/15_111/_pdf
* "Interactive Procedural Modelling of Coherent Waterfall Scenes": https://inria.hal.science/hal-01095858/file/waterfall.pdf
