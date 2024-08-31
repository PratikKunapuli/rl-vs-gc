<h1 align="center">
  <b>AerialManipulation</b><br>
</h1>

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-1.1.0-green.svg)](https://github.com/isaac-sim/IsaacLab/tree/v1.1.0)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)

Repository for exploring Reinforcement Learning (RL) approaches to control of a 2 degree of freedom (DoF) aerial manipulator, written for IsaacLab

# Installation
## Pre-requisites
### Conda environment 
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
```
### IsaacSim
Instructions for installing IsaacSim are reproduced here for clarity - but you should follow the [official instructions](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html) for any troubleshooting. 

<details>
<summary>Python environment installation</summary>

```bash
pip install isaacsim==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```
(Optional)
```bash
pip install isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```
</details>

### IsaacLab
Installations for installing IsaacLab are reproduced here for clarity - but you should follow the [official instructions](https://isaac-sim.github.io/IsaacLab/source/setup/installation/pip_installation.html#installing-isaac-lab) for any troubleshooting. 

<details>
<summary>Installing IsaacLab</summary>

Clone the repo locally
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
```
Install dependencies via `apt`
```bash
sudo apt install cmake build-essential
```
Install the library (this should find the previously created conda env `isaaclab` since it is the default name, but if you changed the name for the conda environment you can specify the env name in this command)
```bash
cd IsaacLab
./isaaclab.sh --install
```
</details>

## AerialManipulation Installation
Clone the repo
```bash
git clone git@github.com:PratikKunapuli/AerialManipulation.git
```

Install locally
```bash
cd AerialManipulation
pip install -e .
```

# Demos
Two demonstration files are available to investigate the model/physics. 

1. `demo_sim.py`: A script used to launch an IsaacSim window where the dynamics are simulated directly 
2. `demo_env.py`: A script used to launch an IsaacSim window from the `gymnasium` api. 

# Environments

# Training

# Tools
<details>
<summary>Converting URDFs to USD</summary>

```bash
python ./IsaacLab/source/standalone/tools/convert_urdf.py ./AerialManipulation/models/aerial_manipulator_2dof.urdf ./AerialManipulation/models/aerial_manipulator_2dof.usd --merge-joints --make-instanceable
```
</details>