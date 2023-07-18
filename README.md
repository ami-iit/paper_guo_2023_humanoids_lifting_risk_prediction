<h1 align="center">
  Online Action Recognition for Human Risk Prediction with Anticipated Haptic Alert via Wearables
</h1>


<div align="center">

Cheng Guo, Lorenzo Rapetti, Kourosh Darvish, Riccardo Grieco, Francesco Draicchio and Daniele Pucci

</div>


<div align="center">

  
https://github.com/ami-iit/paper_Guo_2023_Humanoid_Action_Recognition_For_Risk_Prediction/assets/52885318/91d65383-5a9f-4169-8811-c19fe094e0c1

</div>

<div align="center">
  
2023 IEEE IEEE-RAS International Conference on Humanoid Robots (Humanoids)

</div>


<div align="center">
  <a href="#data"><b>Data</b></a> |
  <a href="#dependencies"><b>Dependencies</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#running"><b>Running</b></a> |
  <a href><b>Paper</b></a> |
  <a href><b>Video</b></a>
</div>

### Data
:unlock: The labeled dataset (as txt files), raw wearables dataset and models can be downloaded [here](https://huggingface.co/datasets/ami-iit/manual_lifting_task_dataset).

### Dependencies
#### :pushpin: Required
- [**YARP**](https://github.com/robotology/yarp): a library and toolkit for communication and device interfaces.
- [**YCM**](https://github.com/robotology/ycm): a set of CMake files that support creation and maintenance of repositories and software packages
- [**CMake**](https://cmake.org/download/): an open-source, cross-platform family of tools designed to build, test and package software.
- [**HDE**](https://github.com/robotology/human-dynamics-estimation/blob/master/README.md): a collection of YARP devices for the online estimation of the kinematics and dynamics of a human subject.
#### :paperclip: Optional
- [**iDynTree**](https://github.com/robotology/idyntree): a library of robots dynamics algorithms for control, estimation and simulation.
- [**Wearables**](https://github.com/robotology/wearables): a library for communication and interfaces with wearable sensors.
- [**iFeel**](https://github.com/ami-iit/component_ifeel): a wearable perception system providing kinematic (position and velocities) and dynamic human information.

:fire: Ubuntu 20.04.5 LTS (Focal Fossa) is used in this project.

### Installation
First download this repository:
```sh
git clone https://github.com/ami-iit/paper_Guo_2023_Humanoid_Action_Recognition_For_Risk_Prediction.git
```
#### :wrench: Install robotology-superbuild
- Install [mamba](https://mamba.readthedocs.io/en/latest/) if you don't have one, you can follow the instructions [here](https://github.com/robotology/robotology-superbuild/blob/master/doc/install-mambaforge.md).

#### :wrench: Install this project





### Running
#### :hammer: Offline annotation and training

#### :hammer: Test on recorded data


#### :hammer: Online inference




### Maintainer

:bust_in_silhouette: This repository is maintained by:

|                                                              |                                                      |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| [<img src="https://github.com/Zweisteine96.png" width="40">](https://github.com/Zweisteine96) | [@Zweisteine96](https://github.com/Zweisteine96) |
