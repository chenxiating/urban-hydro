# urban-hydro 
This repository contains a model that couples gray and green infrastructure for watershed-level monitoring.

**Chen et al. (2023), Integrating the spatial configurations of green and gray infrastructure in urban stormwater networks**

## Code
This code is written in Python 3.8 and requires [SWMM computational engine] (https://www.epa.gov/water-research/storm-water-management-model-swmm) to be installed in order to perform the hydrologic-hydraulic modeling in 'make_SWMM_inp.py'. The location for SWMM engine needs to be specified under method 'record_SWMM' in script 'make_SWMM_inp.py'.
* 'Gibbs.py' - generate spanning trees according to Gibbs distribution. 
    - Input: size of the lattice grid, parameter for flow path meandering, parameter for Gibbs distribution
    - Output: network without attributes
* 'hydro_network.py' - assign network attributes
    - Input: desired stormwater network attributes
    - Output: stormwater networks with attributes
* 'make_SWMM_inp.py' - run SWMM and record results
    - Input: environmental conditions (e.g. rainfall hydrographs)
    - Output: hydrological outcomes (e.g. peak flow, flooding)
* 'run_mp_generate_trees.py' - multiprocessing script to generate spanning trees using 'Gibbs.py'
* 'run_multiprocessing_pool.py' - multiprocessing script to run SWMM simulations using 'make_SWMM_inp.py', without green infrastructure
* 'run_mp_SWMM_coverage.py' - multiprocessing script to run SWMM simulation with changing green infrastructure coverage
* 'run_mp_SWMM_plcmt.py' - multiprocessing script to run SWMM simulation with changing green infrastructure's distance to outlet