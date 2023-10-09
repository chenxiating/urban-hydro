# urban_green_gray_model
This repository contains a model that couples gray and green infrastructure for watershed-level monitoring. By representing green infrastructure as bioretention cells, and gray infrastructure as urban stormwater networks, we can precisely control for various gray-green infrastructure spatial configurations and assess their hydrological outcomes in a urban watershed. 

** Chen et al. (2023), Integrating the spatial configurations of green and gray infrastructure in urban stormwater networks, Water Resources Research (in review) **

## Code
This code is written in Python 3.8 and requires [SWMM computational engine] (https://www.epa.gov/water-research/storm-water-management-model-swmm) to be installed in order to perform the hydrologic-hydraulic modeling in 'make_SWMM_inp.py'. **The location for SWMM engine needs to be defined in line 24 as 'my_swmm_path' in script 'make_SWMM_inp.py'.**

The following three scripts are the main skeletons for the project. 

* 'Gibbs.py' - generate spanning trees according to Gibbs distribution. 
    - Input: size of the lattice grid, parameter for flow path meandering, parameter for Gibbs distribution
    - Output: network without attributes
* 'hydro_network.py' - assign network attributes
    - Input: desired stormwater network attributes
    - Output: stormwater networks with attributes
* 'make_SWMM_inp.py' - run SWMM and record results
    - Input: network and green infrasturcture configurations, environmental conditions (e.g. rainfall hydrographs)
    - Output: hydrological outcomes (e.g. peak flow, flooding)

Double check the default values in 'hydro_network.py' to see if they are set to desirable values. Once you have confirmed, you could set up your network attributes in 'main' and your SWMM running options under method 'record_SWMM' in 'make_SWMM_inp.py'. 

In order to speed up the simulation, the following scripts were written for parallel processing the scripts. 

* 'run_mp_generate_trees.py' - multiprocessing script to generate spanning trees using 'Gibbs.py'. 
* 'run_mp_SWMM_coverage.py' - multiprocessing script to run SWMM simulation with changing green infrastructure coverage. Output will be in the same folder as the networks. 
* 'run_mp_SWMM_plcmt.py' - multiprocessing script to run SWMM simulation with changing green infrastructure's distance to outlet. Output will be in the same folder as the networks. 

**Important notes:** 
1. 'run_mp_generate_trees.py' will need to be run first to generate networks. 
2. **The 'path' parameter needs to be updated in both 'run_mp_SWMM_coverage.py' and 'run_mp_SWMM_plcmt.py'**, based on the networks generated from 'run_mp_generate_trees.py'.
3. Currently, the code is not set up to run 'run_mp_SWMM_coverage.py' and 'run_mp_SWMM_plcmt.py' simultaneously. 

If you have any questions while using this code, please contact the author at chen7090@umn.edu.
