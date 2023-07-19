<h1 align="center">
  Online Action Recognition for Human Risk Prediction with Anticipated Haptic Alert via Wearables
</h1>


<div align="center">

Cheng Guo, Lorenzo Rapetti, Kourosh Darvish, Riccardo Grieco, Francesco Draicchio and Daniele Pucci

</div>


<div align="center">



https://github.com/ami-iit/paper_Guo_2023_Humanoid_Action_Recognition_For_Risk_Prediction/assets/52885318/7200df81-d918-4133-ab9e-b0bae716360e



</div>

<div align="center">
  
2023 IEEE-RAS International Conference on Humanoid Robots (Humanoids)

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
To annotate the data, one may follow the instructions below:
- Launch the [yarpserver](https://www.yarp.it//v3.5/yarpserver.html): 
```
yarpserver --write
```
- Run [yarpdataplayer](https://www.yarp.it/latest/group__yarpdataplayer.html) with: 
```
yarpdataplayer --withExtraTimeCol 2
```
- Go to `~/robotology-superbuild/src/HumanDynamicsEstimation/conf/xml` and run the configuration file (in case full joints list use [`Human.xml`](https://github.com/robotology/human-dynamics-estimation/blob/master/conf/xml/Human.xml), in case reduced joints list use [`HumanStateProvider_ifeel_0.xml`](https://github.com/ami-iit/component_ergocub/blob/main/software/experiments/2022_04_Aereoporti_Roma/conf/HumanStateProvider_ifeel_0.xml)):
```
yarprobotinterface --config proper-configuration-file.xml
```
- Before going to `~/element_human-action-intention-recognition/build/install/bin`, be sure in the  virtual environment previsouly installed, then you may run (make sure all parameters in [humanDataAcquisition.ini](https://github.com/ami-iit/element_human-action-intention-recognition/blob/cheng_CleanUpCode/code/modules/humanMotionDataAcquisition/app/robots/humanDataAcquisition.ini) are set properly):
```
./humanDataAcquisitionModule --from humanDataAcquisition.ini
```
- To start annotation you may need to visualize the human model by running (also be sure the parameters setting in [HumanPredictionVisualizer.ini](https://github.com/ami-iit/element_human-action-intention-recognition/blob/cheng_CleanUpCode/code/modules/humanPredictionVisualizer/app/robots/HumanPredictionVisualizer.ini) are correct):
```
./HumanPredictionVisualizer --from HumanPredictionVisualizer.ini
```
Recalling the index of each action defined [here](https://github.com/ami-iit/element_human-action-intention-recognition/blob/cheng_CleanUpCode/code/modules/humanMotionDataAcquisition/app/robots/humanDataAcquisition.ini#L55), one can annotate the data manually. 
#### :hammer: Test on recorded data
- First of all, make sure [yarpserver](https://www.yarp.it//v3.5/yarpserver.html) is running.
- Open [yarpdataplayer](https://www.yarp.it/latest/group__yarpdataplayer.html) to replay data.
- Go to `~/robotology-superbuild/src/HumanDynamicsEstimation/conf/xml` and run [configuration file](https://github.com/ami-iit/component_ergocub/blob/main/software/experiments/2022_04_Aereoporti_Roma/conf/HumanStateProvider_ifeel_0.xml) (for 31 reduced joints DoF) with:
```
yarprobotinterface --config configuration_file_name.xml
```
- Then go to `~/element_human-action-intention-recognition/build/install/bin` and run:
```
./humanDataAcquisitionModule --from humanDataStreamingOnlineTest.ini
```
- (Remember be in virtual environment) Go to `~/element_human-action-intention-recognition` and run:
```
python3 ./scripts/MoE/main_test_moe.py
```
- (Remember be in virtual environment) Additional: for displaying the action recognition/motion prediction results, go to `~/element_human-action-intention-recognition_modified/scripts/MoE` and run:
```
bash ./runAnimators.sh
```
- Additional: for visualizing simulated human models, go to `~/element_human-action-intention-recognition_modified/build/install/bin` and run:
```
./HumanPredictionVisualizer --from HumanPredictionVisualizer.ini
```
- Additional: in case calibrating the simulated model, download the file [here](https://github.com/ami-iit/component_ergocub/blob/main/software/experiments/2022_04_Aereoporti_Roma/scripts/TPoseCalibration.sh) and run it when human model is in `T-pose`(you can stop the `yarpdataplayer` first when calibrating, afterwards replay it again):
```
bash ./TPoseCalibration.sh zero
```
- Go to `~/element_risk-prediction` and run:
```
python3 ./src/main_model_based_risk_evaluation.py
```
- To start NIOSH-based ergonomics evaluation module, run:
```
python3 ./src/niosh_method/nioshOnlineEasyUse.py
```
- To display ergonomics evaluation results, go to `~/element_risk-prediction/src/niosh_method` and run:
```
bash ./runAnimators.sh
```

#### :hammer: Online inference
Under construction, for the moment one may follow the instructions [here](https://github.com/ami-iit/component_ergocub/issues/145).


### Maintainer

:bust_in_silhouette: This repository is maintained by:

|                                                              |                                                      |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| [<img src="https://github.com/Zweisteine96.png" width="40">](https://github.com/Zweisteine96) | [@Zweisteine96](https://github.com/Zweisteine96) |
