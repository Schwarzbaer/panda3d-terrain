Terrain for Panda3D
===================

Just throwing some academic papers at the GPU to see what sticks.


Current Model
-------------

* Map Data
  * `terrain_height`: Height of the terrain
  * `water_height`: Current height of the water
  * `water_influx`: Water influx rate to simulate e.g. rain, or springs.
* Scalar model parameters
  * `evaporation_constant`: 0.0 for no evaporation. 1.0 for "If the
     rate of evaporation would be constant at the initial frame, all
     water would evaporate within one second."
* Frame parameters
  * `dt`: time step. The amount of time for which the simulation will be
    advanced.
* Intermediate Maps
  * `water_height_after_influx`
  * `water_height_after_crossflux`
  * `water_height_after_evaporation`
* Process
  * Influx of water
    `water_height_after_influx = water_height + water_influx * dt`
  * Evaporate water
    `water_height_after_evaporation = water_height_after_crossflux * (1 - evaporation_constant * dt)`


Papers
------

* "Fast Hydraulic Erosion Simulation and Visualization on GPU": https://xing-mei.github.io/files/erosion.pdf
  * Influx of water
  * Calculating the pressure-based outflux from each cell to its neighbors
  * Update water height
  * Calculate water velocity
  * Erode material from terrain into water, or deposit it
  * Transport sediment through water flow
  * Evaporate water
* Map Data
  * `w`: Water influx rate during this frame
  * `b`: Terrain height
  * `d`: Water height
  * `s`: Suspended sediment amount
  * `f`: Outflow flux
  * `v`: Velocity field
* Scalar model parameters
  * `g`: Gravity
  * `l`: Length of the pipe connecting cells
  * `A`: Cross-section of the pipe connecting cells
  * `Ks`: Dissolving constant
  * `Kd`: Deposition constant
  * `Ke`: Evaporation constant
* Frame parameters
  * `dt`: time step.
* Intermediate Maps
  * `d1`: Water height after the initial influx
  * `d2`:
* Process
  * DONE: Increase water
    Apply desired changes to `w`
    d1 = d + w * dt
  * Compute outflux flow; We treat neighboring cells as if they were connected by pipes.
    Consider boundary conditions! Make flux to edges 0, or loop it around.
    * `cf`: Current flux
    * `dh`: Height difference between cells = (own terrain height + own `d1`) - (neighbor terrain height + neighbor `d1`)
    pipe coefficient = A * (g * dh / l)
    neighbor flux = max(0, current flux + dt * pipe coefficient)
    scaling factor = min(1, `d1` * xy distance / (sum of all flux * dt))
    total flux = scaling factor * all neighbor fluxes
  * Update water surface
    net water volume change = dt * all neighbor fluxes
    new water height `d2` = `d1` + net water volume change / distance to neighors  # WTF???
  * Update velocity field
    Consider only vertical flow.
    `u` / `v` are velocity in X / Y direction
    For X and Y direction separately:
    * delta Water `dWX` = sum up the delta flows with the neighbors.
    * equation: `dWX` = distance Y (???) * (average of `d1` and `d2`) * u
    * derive u (and v accordingly for `dWY`).
    For simulation to be stable, dt * u/v velocity <= distance to neighbor
  * Erosion-deposition
    Note: local tilt angle may require setting a lower bound, otherwise sediment transport capacity will be near zero at low tilt.
    sediment transport capacity `C` = scaling factor * sin(local tilt angle) * |velocity|
    if sediment transport capacity > suspended sediment amount:  # add soil to water
        dissolved amount = Ks * (C - s)
        new terrain height = b - dissolved amount
        intermediate sediment amount `s1` = s + dissolved amount
    else:  # deposit sediment
        deposited amount = Kd * (s - C)
        new terrain height = b + deposited amount
        intermediate sediment amount `s1` = s - deposited amount
  * Transport sediment
    new suspended sediment amount = s1 at position - uv * dt, interpolating the four nearest neighbors
  * DONE: Evaporate water
    Temperature is assumed to be the same everywhere.
    new water height = `d2` + (1 - Ke * dt)
