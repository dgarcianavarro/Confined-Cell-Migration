# Confined-Cell-Migration
A subcellular model developed within the PhysiCell framework for simulating confined cell migration under different levels of confinement, based on an agent-based mechanobiological migration model in confined geometries and incorporating a Bayesian optimization calibration pipeline with Optuna to estimate parameters that reproduce and characterize experimental migration behavior.

Developed by David Garcia-Navarro, University of Zaragoza. E-mail: david.garcian@unizar.es

## Description
Each file in this repository must be placed in its corresponding directory to ensure the correct execution of the PhysiCell workflow and proper handling of simulation inputs and outputs. To run the parameter calibration pipeline, execute Optimization.py, which launches the Bayesian optimization workflow using Optuna. This script automatically evaluates simulation outputs against experimental targets and iteratively updates the parameter space to identify configurations that best reproduce observed confined migration behavior.

## Reference
