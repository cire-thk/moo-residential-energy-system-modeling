# Modeling Residential Energy Systems and Building Renovation using Evolutionary Algorithms with Multi-Objective Optimization
This project includes an energy system model of a multifamily residential including photovoltaic-thermal roof tiles, an air sourced heat-pump, battery electric vehicles, a thermal storage, a battery storage and a gas boiler. Furthermore the sizing of these components can be optimized by a multi-objective optimization using evolutionary algorithms. This work corresponds to this [paper](https://energsustainsoc.biomedcentral.com/). This document serves to provide an overview of the project, with the objective of introducing this simulation for further research.

__Energy System__
<p align= "center"><img src="Energy System.png" width="600"><br/></p> 

## Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Results](#results)
8. [Contribution](#contribution)
9. [Citation](#citation)
10. [License](#license)

## Introduction
The work was established over the course of the master thesis of [Marius Bartkowski](https://www.researchgate.net/profile/Marius-Bartkowski). Further extensions were made by [other students](#contribution) from the [Cologne Universitiy of Applied Sciences](https://www.th-koeln.de/en/). 
The objectives within the multi-objective optimization are the CO2 emissions and annualized total cost of the energy system. The decision variables are listed below:

- Installed electrical power of the western and eastern oriented photovoltaic-thermal plant
- Capacity of Battery electrical energy storage
- Capacity of warm water storage
- Nominal thermal power of the air sourced heat-pump
- Bivalence point
- Type of fuel for gas boiler
- Energy management system for battery electrical vehicle
- Energetic renovation level
- (Installed electrical power of the combined heat and power hydrogen fuel cell)

The multi-objective optimization can also be used for a sensitivity analysis in which a given input parameter is deviated. The energy system can also be disconnected from the optimization for analyzing the energy system at given set of decision varaibles. Furthermore, there are two different energy system configurations. They differ in only one component which is a hydrogen fueled fuel cell. This unit is considered a combined heat and power system.

## Prerequisites
This simulation is run within a python environment and certain libraries are mandatory for running it. Environments can be easily managed with the help of Anaconda.
The following libraries are necessary to include in the environment:

- pandas
- numpy
- matplotlib
- openpyxl
- pvlib
- tqdm
- pymoo
- pandas
- joblib

## Installation
As mentioned before creating a python environment is necessary for running this simulation. Downloading Anaconda as an easy way to manage environments is recommended.
This [link](https://www.anaconda.com/download) will direct you towards it's download. After installing Anaconda you can either choose to work with Anaconda Navigator (GUI) or the Anaconda Prompt (Console).
For additional information, please refer to the various online resources on this subject.

## Quick start
The source code itself is organized in several python files. Their names and their corresponding functionality is listed down below:

- Green_Building_Opt:
    - reading and Shaping Inputdata
    - timeseries simulation and managing
    - calling functions and classes from other python files
    - it is basically the main python file
- energy_system:
    - multiple classes for calling during the timeseries simulation
- plotting_functions:
    - functions for arranging results or executing the plotting of results like the pareto front, parallel coordinate plot and sensitivity analysis
- parametrization:
    - functions for parameter regressions, CO2 emissions and cost calculation
- optimization:
    - executing the optimization
    - raising performance metrics
    - defining decision variables and their respective boundaries
- electrical_vehicle:
    - electrical vehicle energy management system modelling

Alongside these python files there are directories for the input and output data. Inside the input directory there are further directories containing various timeseries from weather to consumption data. There is also an excel file (.xlsx) containing all the intransient inputs. This excel file's worksheets are gathered like this:

- Own Design:
    - providing predefined set of decision variables for a non optimization simulation
- General Parameters:
    - technology unrelated parameters like density and isobaric heat capacity of water
- Optimization Parameters:
    - defining Number of generations, populations as well as evolutionary algorithm
- Technical Parameters:
    - technical parameters of energy system components
- Economical Parameters:
    - economical parameters like energy and component prices
- Ecological Parameters:
    - emission factor for natural gas

The output directory divides itself into two subsequent directories. One is for optimization results and the other is for the timeseries results of a non optimization run.

## Results
The most important results are the Pareto front and the parallel coordinate plot which can be seen below.

__Pareto Front__
<p align= "center"><img src="Output/Optimizations/Illustrations/2024-11-14/ParetoFrontsA_NSGA2_Gen60_Pop50_2024-11-14.png" width="600"><br/></p> 

The Pareto Front is given in the space of feasible solutions and marks the non-dominated solutions. The coloring and shaping of these solutions indicate corresponding decision variables. For example the colours show the level of renovation, the shape indicates the energy carrier and the border coloration indicates whether the energy management system is applied or not.

__Parallel Coordinate Plot__
<p align= "center"><img src="Output/Optimizations/Illustrations/2024-11-14/Parallel Coordinate Plot.png" width="600"><br/></p> 

The parallel coordinate plot gives an overview of every solution with their objective values and decision variables. One line illustrates a particular solution.

## Contribution
The development of this program was assisted by the following people:

- Henrik Naß
- Fares Aoun
- Jannis Grunenberg

## Citation
If you need to reference to this simulation or parts of this simulation, please refer to the corresponding publication:

> Bartkowski et al., (2025). Modeling Residential Energy Systems and Building Renovation using Evolutionary Algorithms with Multi-Objective Optimization. Energ Sustain Soc, x(xx), xxxx, doi


## License
This simulation is copyrighted by Marius Bartkowski and licensed with BSD-3-Clause terms, found [here](https://github.com/cire-thk/BifacialSimu/blob/master/LICENSE).
