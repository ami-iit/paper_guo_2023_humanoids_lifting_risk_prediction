# ===============================================================================
#
# This function implements the NIOSH lifting equation for online risk evaluation.
# Don't use this function now, go to runNoishOnline.py >_<
#
# by Cheng Guo
# ================================================================================
import time
import yarp
import casadi as cs
import numpy as np
import idyntree.bindings as idt
from tabulate import tabulate as table
import nioshUtilities as nut
import nioshConfiguration as ncf
#import idyntree 
#from matplotlib.pylab import *

##############################################
############### Initialization ###############
##############################################
## initialize yarp network ##
yarp.Network.init()
if not yarp.Network.checkNetwork():
    print("[ERROR] Unable to open a YARP Network.")
else:
    print("[INFO] Open a YARP Network happily.")

## initialize yarp ports for receiving data ##
# joint configurations
human_jPos_port = yarp.BufferedPortBottle()
human_jPos_port.open("/risk_prediction/humanJointPos:i")
humanState_is_connected = yarp.Network.connect("/risk_prediction/jointPositions:o", 
                                               "/risk_prediction/humanJointPos:i")
print("[INFO] Read human joint positions port is connected: {}".format(humanState_is_connected))

# base configurations
basePosOrien_port = yarp.BufferedPortBottle()
basePosOrien_port.open("/risk_prediction/base_pos_orien:i")
basePose_is_connected = yarp.Network.connect("/humanDataAcquisition/basePose:o", 
                                             "/risk_prediction/base_pos_orien:i")
print("[INFO] Read base positions/orientations port is connected: {}".format(basePose_is_connected))

# motion predictions (size: 22*3, joint rot positions)
#motion_prediction_port = yarp.BufferedPortBottle()
#motion_prediction_port.open("/risk_prediction/motion_predictions:i")
#motionPrediction_is_connected = yarp.Network.connect("/test_moe/motionPrediction:o", 
#                                                     "/risk_prediction/motion_predictions:i")
#print("[INFO] Read motion predictions port is connected: {}".format(motionPrediction_is_connected))

# dynamic predictions (size: 2*6, foot forces/torques)
#dynamic_prediction_port = yarp.BufferedPortBottle()
#dynamic_prediction_port.open("/risk_prediction/dynamic_predictions:i")
#dynamicPrediction_is_connected = yarp.Network.connect("/test_moe/dynamicPrediction:o", 
#                                                      "/risk_prediction/dynamic_predictions:i")
#print("[INFO] Read dynamic predictions port is connected: {}".format(dynamicPrediction_is_connected))

# action recognitions (size: 25*12, probabilities vector for 12 actions in 25 output steps)
action_recognition_port = yarp.BufferedPortBottle()
action_recognition_port.open("/risk_prediction/actionRecognitions:i")
actionRecognition_is_connected = yarp.Network.connect("/test_moe/actionRecognition:o", 
                                                      "/risk_prediction/actionRecognitions:i")
print("[INFO] Read action recognitions port is connected: {}".format(actionRecognition_is_connected))

## initialize yarp ports for publishing NIOSH results ##
RWL_port = yarp.BufferedPortBottle()
RWL_bottle = RWL_port.prepare()
RWL_port.open("/risk_prediction/recommond_weight_load:o")

RI_port = yarp.BufferedPortBottle()
RI_bottle = RI_port.prepare()
RI_port.open("/risk_prediction/risk_index:o")

NIOSH_paras_port = yarp.BufferedPortBottle()
NIOSH_paras_bottle = NIOSH_paras_port.prepare()
NIOSH_paras_port.open("/risk_prediction/niosh_paras:o")

NIOSH_factors_port = yarp.BufferedPortBottle()
NIOSH_factors_bottle = NIOSH_factors_port.prepare()
NIOSH_factors_port.open("/risk_prediction/niosh_factors:o")

## initialize kinDynComputation object & load human model ##
joint_list = ["jT9T8_rotx",
              "jT9T8_roty",
              "jT9T8_rotz",
              "jRightShoulder_rotx",
              "jRightShoulder_roty",
              "jRightShoulder_rotz",
              "jRightElbow_roty",
              "jRightElbow_rotz",
              "jLeftShoulder_rotx",
              "jLeftShoulder_roty",
              "jLeftShoulder_rotz",
              "jLeftElbow_roty",
              "jLeftElbow_rotz",
              "jLeftHip_rotx",
              "jLeftHip_roty",
              "jLeftHip_rotz",
              "jLeftKnee_roty",
              "jLeftKnee_rotz",
              "jLeftAnkle_rotx",
              "jLeftAnkle_roty",
              "jLeftAnkle_rotz",
              "jLeftBallFoot_roty",
              "jRightHip_rotx",
              "jRightHip_roty",
              "jRightHip_rotz",
              "jRightKnee_roty",
              "jRightKnee_rotz",
              "jRightAnkle_rotx",
              "jRightAnkle_roty",
              "jRightAnkle_rotz",
              "jRightBallFoot_roty"]

urdf_file = "humanModels/humanSubject01_66dof.urdf"
dynCom = idt.KinDynComputations()

model_loader = idt.ModelLoader()
#model_loader.loadModelFromFile(urdf_file)
model_loader.loadReducedModelFromFile(urdf_file, joint_list, "urdf")
#DOF = model_loader.model().getNrOfDOFs()
#print("model dofs is: ", DOF)
model = model_loader.model()
dynCom.loadRobotModel(model_loader.model())
dynCom.setFloatingBase("Pelvis")

name_to_index = [int(model.getJointIndex(joint_name)) for joint_name in  joint_list]

## initialize buffer variables ##
dofs = dynCom.model().getNrOfDOFs()
#print("[INFO] Dofs is: ", dofs)
s = idt.VectorDynSize(dofs)
ds = idt.VectorDynSize(dofs)
H_b = idt.Transform()
w_b = idt.Twist()

## initialize all velocities to zero ##
s.zero()
ds.zero()
w_b.zero()

## initialize gravity vector ##
gravity = idt.Vector3()
gravity.zero()
gravity.setVal(2, -9.81)

## initialize the niosh object ##
niosh = nut.nioshOnlineInference()

# initialize other parameters
count = 0
weight_load = 5
#action_label_origin = "none"

link_L5S1_posx0 = 0
link_LeftHand_posx0 = 0
#link_LeftHand_posz0 = 0
link_LeftFoot_posz0 = 0

#link_LeftHand_posx0_ref = 0
#link_LeftHand_posz0_ref = 0

################################################
############### Online Inference ###############
################################################
while True:
    # read human joint rot positions
    read_human_jRotPos = human_jPos_port.read()
    if read_human_jRotPos is not None:
        human_jPos_data = []
        for i in range(read_human_jRotPos.size()):
            human_jPos_data.append(read_human_jRotPos.get(i).asFloat64())
    #print("[INFO] Read human joint positions: ", human_jPos_data)
    print('================================================')
    print('======== Configuration for Iteration {} ========'.format(count))
    print('================================================')
    print("[INFO] Read human joint positions successfully!")

    # read base positions and orientations
    read_base_posOrien = basePosOrien_port.read()
    if read_base_posOrien is not None:
        base_pos = []
        base_orien = []
        base_orien_pos = []
        for i in range(read_base_posOrien.size()):
            if i < 3:
                # the first three elements are base positions
                base_pos.append(read_base_posOrien.get(i).asFloat64())
            else:
                # the last four are base orientations
                base_orien.append(read_base_posOrien.get(i).asFloat64())
        base_orien_pos = np.concatenate((base_orien, base_pos))
    #print("[INFO] Read base positions: ", base_pos)
    #print("[INFO] Read base orientations: ", base_orien)
    print("[INFO] Read base configurations successfully!")
    
    # read action recognition probabilities
    read_action_recognition = action_recognition_port.read()
    if read_action_recognition is not None:
        action_recognitions = []
        for i in range(read_action_recognition.size()):
            action_recognitions.append(read_action_recognition.get(i).asFloat64())
    #print("[INFO] Read action recognitions: ", action_recognitions)
    print("[INFO] Read action recognitions successfully!")

    # convert to 25*3 numpy array
    action_array = np.array(action_recognitions).reshape((25, 3))
    # get only the probability vector of first output step 
    action_prediction_array = action_array[5, :]
    # find the action label with higest probability
    #action_index = np.where(action_prediction_array == max(action_prediction_array))
    action_index_now = action_prediction_array.tolist().index(max(action_prediction_array))
    action_label_now = ncf.action_labels[action_index_now]
    
    # update joint configuration
    #print(name_to_index)
    #print("name index len"+str(len(name_to_index) ))
    #print("joint list"+str(len(joint_list)))
    #print("human jpos"+str(len(human_jPos_data )))    
    #print(["" if name_to_index[i] != -1 else joint_list[i] for i in range(len(human_jPos_data))])

    for i in range(len(human_jPos_data)):
        #s[name_to_index[i]] = human_jPos_data[i]
        s.setVal(name_to_index[i], human_jPos_data[i])
    #print("[INFO] Size of human joints data is: ", np.size(human_jPos_data))
    #s_np = human_jPos_data
    #s = idt.VectorDynSize.FromPython(s_np)
    #print("s size: ", s.size())

    # update base configuration
    H_b_np = nut.Quaternion2Mat()(base_orien_pos)
    H_b.fromHomogeneousTransform(idt.Matrix4x4_FromPython(H_b_np))

    # update the kinDynComputation object with the new configuration
    dynCom.setRobotState(H_b, s, w_b, ds, gravity)

    #base_pos_vector = np.array([[base_pos[0]],
    #                            [base_pos[1]],
    #                            [base_pos[2]]])
    
    # /////////////////////////////////////////////////////////
    # compute rotation matrix
    #rotM = Quaternion2RotationMat(base_orien)
    # /////////////////////////////////////////////////////////

    # retrieve rot positions of jL5S1, jHand and jFoot
    #jL5S1_rotPos = np.array([[human_jPos_data[0]],
    #                         [human_jPos_data[25]],
    #                         [human_jPos_data[26]]])
    #jRightWrist_rotPos = np.array([dynCom  [human_jPos_data[10]]])                       
    
    # /////////////////////////////////////////////////////////
    # compute joint positions in world frame
    #jL5S1_pos = forward_Kinematics(base_pos_vector, rotM, jL5S1_rotPos)
    #print("[INFO] Position of jL5S1 is: ", jL5S1_pos)
    #jRightWrist_pos = forward_Kinematics(base_pos_vector, rotM, jRightWrist_rotPos)
    #print("[INFO] Position of jRightWrist is: ", jRightWrist_pos)
    #jLeftAnkle_pos = forward_Kinematics(base_pos_vector, rotM, jLeftAnkle_rotPos)
    #print("[INFO] Position of jLeftAnkle is: ", jLeftAnkle_pos)
    # /////////////////////////////////////////////////////////
    
    # get the desired link configuration w.r.t. world frame
    H_link_L5S1 = dynCom.getWorldTransform("L5")
    H_link_L5S1_np = H_link_L5S1.asHomogeneousTransform().toNumPy()
    link_L5S1_pos = H_link_L5S1_np[:3, 3]
    #print("[INFO] H_link_L5S1 is: ", H_link_L5S1.toString())
    #print("[INFO] H_link_L5S1 is: ", H_link_L5S1_np)
    #print("[INFO] Position of L5S1 in world frame is: ", link_L5S1_pos)

    H_link_LeftFoot = dynCom.getWorldTransform("LeftFoot")
    H_link_LeftFoot_np = H_link_LeftFoot.asHomogeneousTransform().toNumPy()
    link_LeftFoot_pos = H_link_LeftFoot_np[:3, 3]
    #print("[INFO] Position of LeftFoot in world frame is: ", link_LeftFoot_pos)

    H_link_RightFoot = dynCom.getWorldTransform("RightFoot")
    H_link_RightFoot_np = H_link_RightFoot.asHomogeneousTransform().toNumPy()
    link_RightFoot_pos = H_link_RightFoot_np[:3, 3]

    H_link_LeftHand = dynCom.getWorldTransform("LeftHand")
    H_link_LeftHand_np = H_link_LeftHand.asHomogeneousTransform().toNumPy()
    link_LeftHand_pos = H_link_LeftHand_np[:3, 3]
    #print("[INFO] Position of LeftHand in world frame is: ", link_LeftHand_pos)

    H_link_RightHand = dynCom.getWorldTransform("RightHand")
    H_link_RightHand_np = H_link_RightHand.asHomogeneousTransform().toNumPy()
    link_RightHand_pos = H_link_RightHand_np[:3, 3]
    
    ## right/left hands w.r.t pelvis frame
    H_link_lh_ref = dynCom.getRelativeTransform("L5", "LeftHand")
    H_link_lh_ref_np = H_link_lh_ref.asHomogeneousTransform().toNumPy()
    link_lh_pos_ref = H_link_lh_ref_np[:3, 3]

    H_link_rh_ref = dynCom.getRelativeTransform("L5", "RightHand")
    H_link_rh_ref_np = H_link_rh_ref.asHomogeneousTransform().toNumPy()
    link_rh_pos_ref = H_link_rh_ref_np[:3, 3]

    # get the link configuration w.r.t. reference frame
    H_link_LeftHand_ref = dynCom.getRelativeTransform("LeftFoot", "LeftHand") # 'LeftFoot' is reference, 'LeftHand' is objective
    H_link_LeftHand_np_ref = H_link_LeftHand.asHomogeneousTransform().toNumPy()
    link_LeftHand_pos_ref = H_link_LeftHand_np_ref[:3, 3]
    #print("[INFO] Position of LeftHand in LeftFoot frame is: ", link_LeftHand_pos_ref)

    H_link_RightHand_ref = dynCom.getRelativeTransform("RightFoot", "RightHand")
    H_link_RightHand_np_ref = H_link_RightHand.asHomogeneousTransform().toNumPy()
    link_RightHand_pos_ref = H_link_RightHand_np_ref[:3, 3]

    print("[INFO] Desired link configuration is retrieved below: ")
    print("--------------------------------------------------")
    print(table([['L5S1 in world frame', link_L5S1_pos], 
                 ['LeftFoot in world frame', link_LeftFoot_pos],
                 ['LeftHand in world frame', link_LeftHand_pos],
                 ['RightFoot in world frame', link_RightFoot_pos], 
                 ['RightHand in world frame', link_RightHand_pos],
                 ['LeftHand in LeftFoot frame', link_LeftHand_pos_ref],
                 ['RightHand in RightFoot frame', link_RightHand_pos_ref]], 
                 headers=['Description', 'Position']))

    if count == 0:
        action_label_origin = action_label_now
        # just use current joint positions for initialization
        # !need further modification, actually should set as initial status
        #link_LeftHand_posx0_ref = link_LeftHand_pos_ref[0]
        #link_RightHand_posx0_ref = link_RightHand_pos_ref[0]
        #link_Hand_posx0_ref = 0.09 + (link_LeftHand_posx0_ref + link_RightHand_posx0_ref) / 2 
        link_lh_posx0_ref = link_lh_pos_ref[0]
        link_rh_posx0_ref = link_rh_pos_ref[0]
        link_hands_posx0_ref = (link_lh_posx0_ref + link_rh_posx0_ref) / 2 

        link_LeftHand_posz0_ref = link_LeftHand_pos_ref[2]
        link_RightHand_posz0_ref = link_RightHand_pos_ref[2]
        link_Hand_posz0_ref = 0.078 + (link_LeftHand_posz0_ref + link_RightHand_posz0_ref) / 2 

        link_LeftHand_posz0 = link_LeftHand_pos[2]
        link_RightHand_posz0 = link_RightHand_pos[2]
        link_Hand_posz0 = (link_LeftHand_posz0 + link_RightHand_posz0) / 2
    
    # check if any change of human action
    if action_label_now is not action_label_origin:
        current_time = time.perf_counter()
        print("[INFO] Detect human action is changing at time: {}".format(current_time))
        print("[INFO] Current human action is: ", action_label_now, ", previous action is: ", action_label_origin)
        # update action label and origin status
        action_label_origin = action_label_now
        #link_L5S1_posx0 = link_L5S1_pos[0]
        #link_LeftHand_posx0 = link_LeftHand_pos[0]
        #link_LeftHand_posx0_ref = link_LeftHand_pos_ref[0]
        #link_RightHand_posx0_ref = link_RightHand_pos_ref[0]
        #link_Hand_posx0_ref = 0.09 + (link_LeftHand_posx0_ref + link_RightHand_posx0_ref) / 2 
        link_lh_posx0_ref = link_lh_pos_ref[0]
        link_rh_posx0_ref = link_rh_pos_ref[0]
        link_hands_posx0_ref = (link_lh_posx0_ref + link_rh_posx0_ref) / 2

        link_LeftHand_posz0_ref = link_LeftHand_pos_ref[2]
        link_RightHand_posz0_ref = link_RightHand_pos_ref[2]
        link_Hand_posz0_ref = 0.078 + (link_LeftHand_posz0_ref + link_RightHand_posz0_ref) / 2 

        link_LeftHand_posz0 = link_LeftHand_pos[2]
        link_RightHand_posz0 = link_RightHand_pos[2]
        link_Hand_posz0 = (link_LeftHand_posz0 + link_RightHand_posz0) / 2
        #link_LeftFoot_posz0 = link_LeftFoot_pos[2] 
    else:
        print("[INFO] Origin human action is: ", action_label_origin)
        print("[INFO] Current human action is: ", action_label_now)
    
    NIOSH_paras_bottle = NIOSH_paras_port.prepare()
    NIOSH_paras_bottle.clear()

    NIOSH_factors_bottle = NIOSH_factors_port.prepare()
    NIOSH_factors_bottle.clear()

    log_time = time.perf_counter()
    NIOSH_paras_bottle.addFloat64(log_time)
    NIOSH_factors_bottle.addFloat64(log_time)
    # retrieve niosh equation multipliers
    # horizontal distance is determined at the origin moment of an action
    #H = niosh.findH(link_L5S1_posx0, link_LeftHand_posx0)
    H = niosh.findH(0, link_hands_posx0_ref)
    NIOSH_paras_bottle.addFloat64(H)
    HM = niosh.get_HM(H)
    NIOSH_factors_bottle.addFloat64(HM)
    #print("[INFO] H is: ", H, "cm")
    
    # vertical distance is determined at the origin moment of an action
    #V = niosh.findV(link_LeftHand_posz0, link_LeftFoot_posz0)
    V = niosh.findH(link_Hand_posz0_ref, 0)
    NIOSH_paras_bottle.addFloat64(V)
    VM = niosh.get_VM(V)
    NIOSH_factors_bottle.addFloat64(VM)
    #print("[INFO] V is: ", V, "cm")
    
    # vertical lifting distance changes according to the origin/end moments of an action
    link_handPos_now = (link_LeftHand_pos[2] + link_RightHand_pos[2]) / 2
    D = niosh.findD(link_Hand_posz0, link_handPos_now)
    NIOSH_paras_bottle.addFloat64(D)
    DM = niosh.get_DM(D)
    NIOSH_factors_bottle.addFloat64(DM)
    #print("[INFO] D is: ", D, "cm")
    
    # publish [time, H, V, D] to yarp port
    NIOSH_paras_port.write()

    # assume there is no body twist, A is set default as 0
    AM = niosh.get_AM()
    NIOSH_factors_bottle.addFloat64(AM)
    # assume coupling quality is 'Fair'
    CM = niosh.get_CM(V, 'Fair')
    NIOSH_factors_bottle.addFloat64(CM)
    # assume lifting frequency is 3 lifts/min
    # ! actually should calculate according to the data
    FM = niosh.get_FM(V, duration=1)
    NIOSH_factors_bottle.addFloat64(FM)

    # publish [time, HM, VM, DM, AM, CM, FM] to yarp port
    NIOSH_factors_port.write()

    # publish RWL and RI to yarp ports
    RWL_bottle = RWL_port.prepare()
    RWL_bottle.clear()
    RWL = niosh.get_RWL(HM, VM, DM, AM, CM, FM)
    RWL_bottle.addFloat64(RWL)
    RWL_port.write()
    #print("[INFO] RWl is: ", RWL, "kg")

    RI_bottle = RI_port.prepare()
    RI_bottle.clear()
    #if action_label_now is 'none' or 'standing':
    #    # !for lifting-unrealted actions, set RI as 0
    #    RI = 0
    #else:
        # only consider lifting-related actions
    #    RI = niosh.get_RI(weight_load, RWL)
    if action_label_now in ncf.lifting_list:
        RI = niosh.get_RI(weight_load, RWL)
    else:
        RI = niosh.get_RI(0, RWL)
    RI_bottle.addFloat64(RI)
    RI_port.write()

    print('========================================================')
    print('======== NIOSH Online Inference at Iteration {} ========'.format(count))
    print('========================================================')

    # check if any change of human action
    if action_label_now is not action_label_origin:
        current_time = time.perf_counter()
        print("[INFO] Detect human action is changing at time: {}".format(current_time))
        print("[INFO] Current human action is: ", action_label_now, ", previous action is: ", action_label_origin)
        # update action label and origin status
        action_label_origin = action_label_now
        #link_L5S1_posx0 = link_L5S1_pos[0]
        #link_LeftHand_posx0 = link_LeftHand_pos[0]
        #link_LeftHand_posx0_ref = link_LeftHand_pos_ref[0]
        #link_RightHand_posx0_ref = link_RightHand_pos_ref[0]
        #link_Hand_posx0_ref = 0.09 + (link_LeftHand_posx0_ref + link_RightHand_posx0_ref) / 2 
        link_lh_posx0_ref = link_lh_pos_ref[0]
        link_rh_posx0_ref = link_rh_pos_ref[0]
        link_hands_posx0_ref = (link_lh_posx0_ref + link_rh_posx0_ref) / 2

        link_LeftHand_posz0_ref = link_LeftHand_pos_ref[2]
        link_RightHand_posz0_ref = link_RightHand_pos_ref[2]
        link_Hand_posz0_ref = 0.078 + (link_LeftHand_posz0_ref + link_RightHand_posz0_ref) / 2 

        link_LeftHand_posz0 = link_LeftHand_pos[2]
        link_RightHand_posz0 = link_RightHand_pos[2]
        link_Hand_posz0 = (link_LeftHand_posz0 + link_RightHand_posz0) / 2
        #link_LeftFoot_posz0 = link_LeftFoot_pos[2] 
    else:
        print("[INFO] Origin human action is: ", action_label_origin)
        print("[INFO] Current human action is: ", action_label_now)

    #print("[INFO] Current iteration is: ", count) 
    #print("--------------------------------------------------------")
    #print(table([['Weight lifted (kg)', weight_load], ['Horizontal distance (cm)', H], ['Vertical distance (cm)', V],
    #             ['Vertical travel distance (cm)', D], ['Lifting frequency (times/min)', niosh.F], ['Task duration (hours)', 1],
    #             ['Twisting angle (degree)', niosh.A], ['Coupling', 'Fair']],
    #             headers=['Description', 'Iteration {}'.format(count)]))
    #print("--------------------------------------------------------")
    #print(table([['Lifting constant (LC)', niosh.LC], ['Horizontal multiplier (HM)', HM],
    #             ['Vertical multiplier (VM)', VM], ['Distance multiplier (DM)', DM],
    #             ['Frequency multiplier (FM)', FM], ['Twisting multiplier (AM)', AM],
    #             ['Coupling multiplier (CM)', CM], ['Recommended weight load (RWL)', RWL], ['Risk index (RI)', RI]], 
    #             headers=['NIOSH multipliers', 'Iteration {}'.format(count)]))
    #print("--------------------------------------------------------")

    # lifting index guidelines
    #if RI <= 1.0:
    #    print("[INFO] The lift can be considered for most individuals.")
    #elif 1.0 < RI <= 1.5:
    #    print("[INFO] The lift should be evaluated and changed but the urgency is less.")
    #elif 1.5 < RI <= 2.9:
    #    print("[INFO] Modification should be made to reduce the hazard.")
    #elif RI > 2.9:
    #    print("[INFO] The lift is of high risk and must be changed immediately.")

    # update for next iteration
    count += 1
    


