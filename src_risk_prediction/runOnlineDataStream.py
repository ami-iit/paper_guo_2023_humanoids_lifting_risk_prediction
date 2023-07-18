# ===========================================================================
#
# This function implements a model-based online risk prediction tool:
#
# - Read human kinematics/dynamics states from HDE project;
#   - joint positions/velocities
#   - joint torques
# - Read human action recognition & motion prediction from MoE project;
#   - action recognition
#   - motion prediction
#   - origin time of each single action
# - Evaluate lifitng risk level via biomechanical limits;
#   - data processing + animation + human model visualization
# - Evaluate lifting risk level via NIOSH equation;
#   - data processing + animation + human model visualization 
#
# by Cheng Guo
# ============================================================================
from operator import truediv
import numpy as np
from array import *
import yarp 
import hdeModulePythonWrapper as pwr
import mainUtilities as mut
    
if __name__ == "__main__":
    # Configuration
    labels = ["rising", "squatting", "standing"]
    number_categories = len(labels)
    useHumanStateMethods = True
    useHumanDynamicsMethods = False
    firstIteration = True
    count = 1

    # launch the yarp network
    yarp.Network.init()

    if not yarp.Network.checkNetwork():
        print("[ERROR] Unable to open a YARP Network.")
    else:
        print("[INFO] Open a YARP Network happily.")

    # open a yarp resource finder
    rf = yarp.ResourceFinder()

    # human states data
    jointNum = 0
    jointNames = []
    
    jointPositions = []
    jointVelocities = []

    basePositions = []
    baseOrientations = []
    
    # human dynamics data
    jointTorques = []
    
    # initialize yarp ports for publishing data
    joint_torques_port = yarp.BufferedPortBottle()
    joint_torques_bottle = joint_torques_port.prepare()
    joint_torques_port.open("/risk_prediction/jointTorques:o")

    joint_positions_port = yarp.BufferedPortBottle()
    joint_positions_bottle = joint_positions_port.prepare()
    joint_positions_port.open("/risk_prediction/jointPositions:o")

    base_orientations_port = yarp.BufferedPortBottle()
    base_orientations_bottle = base_orientations_port.prepare()
    base_orientations_port.open("/risk_prediction/baseOrientations:o")

    base_positions_port = yarp.BufferedPortBottle()
    base_positions_bottle = base_positions_port.prepare()
    base_positions_port.open("/risk_prediction/basePositions:o")

    
    ## *****************************************************************
    ## yarp remapper devices preparation
    if useHumanStateMethods:
        # parse human states data in yarp port via humanStatesRemapper
        humanStateOptions = yarp.Property()
        humanStateOptions.fromString("(humanStateKeyPort /HDE/HumanStateWrapper/state:o)")
        # associate the key "device" with value
        # after association find(key).asString() will return the string value
        humanStateOptions.put("device", "human_state_remapper")
        # create a myModule instance to complete the configuration
        humanStateDataPort = "humanStateDataPort"
        
        humanStateModule = mut.confModule("humanStateKeyPort", humanStateDataPort)
        humanStateModule.check(humanStateOptions, "humanStateKeyPort")
        humanStateModule.put(humanStateOptions, humanStateDataPort)

        # create an instance of device via polyDriver
        humanStateDevice = yarp.PolyDriver()

        # connect human state remapper device
        if not humanStateDevice.open(humanStateOptions):
            print("[ERROR] Failed to connect human state remapper device.")
        elif firstIteration:
            print("[INFO] Successfully connect human state remapper device.")

        # view the human state interfaces methods 
        humanStateMethods = pwr.viewIHumanState(humanStateDevice)
        if not humanStateMethods or not humanStateDevice:
            print("[ERROR] Failed to access IHumanState Interfaces.")
        elif firstIteration: 
            print("[INFO] Now IHumanState Interfaces are available.")


    if useHumanDynamicsMethods:
        # parse human dynamics data in yarp port via humanDynamicsRemapper
        humanDynamicsOptions = yarp.Property()
        humanDynamicsOptions.fromString("(humanDynamicsKeyPort /HDE/HumanDynamicsWrapper/torques:o)")

        humanDynamicsOptions.put("device", "human_dynamics_remapper")
        humanDynamicsDataPort = "humanDynamicsDataPort"

        humanDynamicsModule = mut.confModule("humanDynamicsKeyPort", humanDynamicsDataPort)
        humanDynamicsModule.check(humanDynamicsOptions, "humanDynamicsKeyPort")
        humanDynamicsModule.put(humanDynamicsOptions, humanDynamicsDataPort)

        humanDynamicsDevice = yarp.PolyDriver()
        # connect human dynamics remapper device
        if not humanDynamicsDevice.open(humanDynamicsOptions):
            print("[ERROR] Failed to connect human dynamics remapper device.")
        elif firstIteration:
            print("[INFO] Successfully connect human dynamics remapper device.")

        # view the human dynamics interfaces methods 
        humanDynamicsMethods = pwr.viewIHumanDynamics(humanDynamicsDevice)
        if not humanDynamicsMethods or not humanDynamicsDevice:
            print("[ERROR] Failed to access IHumanDynamics Interfaces.")
        elif firstIteration: 
            print("[INFO] Now IHumanDynamics Interfaces are available.")

    ## *********************************************************

    ## ******************************************************
    ## Data I/O: MoE part
    # read data from MoE action recognition port
    human_action_port = yarp.BufferedPortBottle()
    human_action_port.open("/risk_prediction/humanAction:i")
    action_is_connected = yarp.Network.connect("/test_moe/actionRecognition:o", "/risk_prediction/humanAction:i")
    print("MoE human action port is connected: {}".format(action_is_connected))

    # read data from MoE motion prediction port
    human_motion_port = yarp.BufferedPortBottle()
    human_motion_port.open("/risk_prediction/humanMotion:i")
    motion_is_connected = yarp.Network.connect("/test_moe/motionPrediction:o", "/risk_prediction/humanMotion:i")
    print("MoE human motion port is connected: {}".format(motion_is_connected))

    yarp.delay(0.5)

    while True:
        
        # read human actions prediction probabilities
        predicted_human_actions = human_action_port.read()
        if predicted_human_actions is not None:
            predicted_human_actions_data = []
            for i in range(predicted_human_actions.size()):
                predicted_human_actions_data.append(predicted_human_actions.get(i).asFloat64())

        predicted_actions_reshaped = np.reshape(predicted_human_actions_data, (-1, number_categories))
        # initialize the action prediction probabilities
        action_prob_standing = np.zeros(0)
        action_prob_squatting = np.zeros(0)
        action_prob_rising = np.zeros(0)
    
        # update the action prediction probabilities
        action_prob_standing = np.append(action_prob_standing, predicted_actions_reshaped[:, 0])
        action_prob_squatting = np.append(action_prob_squatting, predicted_actions_reshaped[:, 1])
        action_prob_rising = np.append(action_prob_rising, predicted_actions_reshaped[:, 2])
        ## *************************************************************************
        
        ## ***********************************************************************
        ## Data I/O: human states & dynamics part
        print("-------------------------------------------")
        jointNum = humanStateMethods.getNumberOfJoints()
        print("[INFO] Joint Numbers: ", str(jointNum))
        print("-------------------------------------------")

        jointNames = humanStateMethods.getJointNames()
        print("[INFO] Joint Names: ", jointNames)
        print("-------------------------------------------")

        # //////////////////////////////////////////////////
        # Not Done: get floating base positions and orientations and publish them
        # Currently reading base configurations from '/humanDataAcquisition/basepose:o' port
        #base_positions_bottle = base_positions_port.prepare()
        #base_positions_bottle.clear()
        basePos = humanStateMethods.getBasePosition()
        #for i in range(np.size(basePos)):
        #    base_positions_bottle.addFloat64(basePos[i])
        #base_positions_port.write()

        basePositions.append(basePos)
        print("[INFO] Human base Position at "+ str(count) + "-th iteration: ", basePos)
        
        #print("[INFO] Size of basePos: ", dArray3.size())
        print("-------------------------------------------")

        #base_orientations_bottle = base_orientations_port.prepare()
        #base_orientations_bottle.clear()
        baseOrien = humanStateMethods.getBaseOrientation()
        #for i in range(np.size(baseOrien)):
        #    base_orientations_bottle.addFloat64(baseOrien[i])
        #base_orientations_port.write()

        baseOrientations.append(baseOrien)
        print("[INFO] Human base Orientation at "+ str(count) + "-th iteration: ", baseOrien)
        #print("[INFO] Type of base orientation: ", type(baseOrien))
        print("-------------------------------------------")
        
        # //////////////////////////////////////////////////
        # publish joint positions to yarp network
        joint_positions_bottle = joint_positions_port.prepare()
        joint_positions_bottle.clear()
        jPos = humanStateMethods.getJointPositions()
        for i in range(np.size(jPos)):
            joint_positions_bottle.addFloat64(jPos[i])
        joint_positions_port.write()

        jointPositions.append(jPos)
        print("[INFO] Joint Positions at "+ str(count) + "-th iteration: ", jPos)
        #print("[INFO] Type of joint positions: ", type(jPos))
        print("-------------------------------------------")

        jVel = humanStateMethods.getJointVelocities()
        jointVelocities.append(jVel)
        print("[INFO] Joint Velocities at " + str(count) + "-th iteration: ", jVel)
        print("-------------------------------------------")
        
        # //////////////////////////////////////////////////
        # publish human joint torques to yarp network
        if useHumanDynamicsMethods:
            joint_torques_bottle = joint_torques_port.prepare()
            joint_torques_bottle.clear()
            jTorq = humanDynamicsMethods.getJointTorques()
            for i in range(np.size(jTorq)):
                joint_torques_bottle.addFloat64(jTorq[i])
            joint_torques_port.write()
 
            jointTorques.append(jTorq)
            print("[INFO] Joint Torques at " + str(count) + "-th iteration: ", jTorq)
            print("-------------------------------------------")
        # //////////////////////////////////////////////////
        print("===========================================")

        count += 1
        firstIteration = False

    # save human state/dynamics data
    fileName = "humanData.txt"
    mut.logData(fileName, jointNames, jointPositions, jointVelocities, jointTorques)
    

