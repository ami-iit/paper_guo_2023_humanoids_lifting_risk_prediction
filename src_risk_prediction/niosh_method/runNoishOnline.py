## Update version of nioshOnlineInference.py
import time
import yarp 
import casadi as cs
import numpy as np
import idyntree.bindings as idt
from tabulate import tabulate as table
import nioshUtilities as nut
import nioshConfiguration as ncf
import idyntree

#### Initialization ####
# open yarp network
yarp.Network.init()
if not yarp.Network.checkNetwork():
    print("[ERROR] Unable to open a YARP Network.")
else:
    print("[INFO] Open a YARP Network happily.")

## open yarp ports for receiving
# joint configurations (output port defined in runOnlineDataStream.py)
human_jPos_port = yarp.BufferedPortBottle()
human_jPos_port.open("/risk_prediction/humanJointPos:i")
humanState_is_connected = yarp.Network.connect("/risk_prediction/jointPositions:o", 
                                               "/risk_prediction/humanJointPos:i")
print("[INFO] Read human joint positions port is connected: {}".format(humanState_is_connected))

# !! base configurations (output port defined in humanDataAcquisition.ini)
basePosOrien_port = yarp.BufferedPortBottle()
basePosOrien_port.open("/risk_prediction/base_pos_orien:i")
basePose_is_connected = yarp.Network.connect("/humanDataAcquisition/basePose:o", 
                                             "/risk_prediction/base_pos_orien:i")
print("[INFO] Read base positions/orientations port is connected: {}".format(basePose_is_connected))

# action recognitions (size: 25*12, probabilities vector for 12 actions in 25 output steps)
action_recognition_port = yarp.BufferedPortBottle()
action_recognition_port.open("/risk_prediction/actionRecognitions:i")
actionRecognition_is_connected = yarp.Network.connect("/test_moe/actionRecognition:o", 
                                                      "/risk_prediction/actionRecognitions:i")
print("[INFO] Read action recognitions port is connected: {}".format(actionRecognition_is_connected))

# motion predictions
motion_prediction_port = yarp.BufferedPortBottle()
motion_prediction_port.open("/risk_prediction/motionPredictions:i")
motionPrediction_is_connected = yarp.Network.connect("/test_moe/motionPredictionAll:o", 
                                                      "/risk_prediction/motionPredictions:i")
print("[INFO] Read all future motion predictions port is connected: {}".format(motionPrediction_is_connected))

## open yarp ports for publishing
RWL_port = yarp.BufferedPortBottle()
RWL_port.open("/risk_prediction/recommond_weight_load:o")

RI_port = yarp.BufferedPortBottle()
RI_port.open("/risk_prediction/risk_index:o")

# send feedbacks to ifeel actuator regarding the risk level
command_ifeel_port = yarp.BufferedPortBottle()
command_ifeel_port.open("/risk_prediction/ifeel_command:o")

NIOSH_paras_port = yarp.BufferedPortBottle()
NIOSH_paras_port.open("/risk_prediction/niosh_paras:o")

NIOSH_factors_port = yarp.BufferedPortBottle()
NIOSH_factors_port.open("/risk_prediction/niosh_factors:o")

# create a temporary port for ergocub demo
ergocub_demo_port = yarp.BufferedPortBottle()
ergocub_demo_port.open("/risk_prediction/ergocub_demo:o")

# kinDynComputation object & human urdf model
# reduced joint list with 31 DoFs
joint_list = ncf.joint_list

urdf_file = "human_model_urdf/humanSubject03_66dof.urdf"
dynCom = idt.KinDynComputations()

model_loader = idt.ModelLoader()
model_loader.loadReducedModelFromFile(urdf_file, joint_list, "urdf")
model = model_loader.model()
dynCom.loadRobotModel(model_loader.model())
# base link
dynCom.setFloatingBase("Pelvis")

# mapping the reduced joint list with full joint list
name_to_index = [int(model.getJointIndex(joint_name)) for joint_name in  joint_list]

## initialize buffer variables ##
dofs = dynCom.model().getNrOfDOFs()
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
action_label_origin = None
action_threshold = 0.25 # threshold of probability change to detect action
print("============================================")
print("[INFO] Please check current payload weight: ")
print("============================================")
weight_load = float(input())
#weight_load = ncf.weight # offline 5, online 7 or 15

link_L5S1_posx0 = 0
link_LeftHand_posx0 = 0
link_LeftFoot_posz0 = 0

freeze_old_action_array = []
#### Online Inference ####
while True:
    # read human joint rot positions
    read_human_jRotPos = human_jPos_port.read()
    if read_human_jRotPos is not None:
        human_jPos_data = []
        for i in range(read_human_jRotPos.size()):
            human_jPos_data.append(read_human_jRotPos.get(i).asFloat64())
    print("[INFO] Read human joint positions successfully!")

    # read base positions and orientations
    read_base_posOrien = basePosOrien_port.read()
    if read_base_posOrien is not None:
        base_pos, base_orien, base_orien_pos = ([] for i in range(3))

        for i in range(read_base_posOrien.size()):
            if i < 3:
                # the first three elements are base positions
                base_pos.append(read_base_posOrien.get(i).asFloat64())
            else:
                # the last four are base orientations
                base_orien.append(read_base_posOrien.get(i).asFloat64())
        base_orien_pos = np.concatenate((base_orien, base_pos))
    print("[INFO] Read base configurations successfully!")

    # read current action recognition probabilities
    read_action_recognition = action_recognition_port.read()
    if read_action_recognition is not None:
        action_recognitions = []
        for i in range(read_action_recognition.size()):
            action_recognitions.append(read_action_recognition.get(i).asFloat64())
    print("[INFO] Read action recognitions successfully!")
    action_array = np.array(action_recognitions).reshape((50, 3))

    # read motion predictions
    read_motion_prediction = motion_prediction_port.read()
    if read_motion_prediction is not None:
        motion_predictions = []
        for i in range(read_motion_prediction.size()):
            motion_predictions.append(read_motion_prediction.get(i).asFloat64())
    print("[INFO] Read motion predictions successfully!")
    motion_array = np.array(motion_predictions).reshape((50, 31))
    # print("[Attention] Motion array shape: ", motion_array[0, :].shape)
    # collect future 0.8s motion predictions (0.8/0.01=80)
    # !now: max 50df, so 20df is 20*0.03=0.6s
    motion_prediction_array = motion_array[0:31, :]
    # print("[Attention] Motion array size: ", len(motion_prediction_array))

    # initialize links list 
    links_lh_world, links_rh_world, links_lh_L5, links_rh_L5 = ([] for i in range(4))

    if ncf.use_motion_predictions: # links list of size 20 (prediction steps)
        print("[INFO] Use motion predictions to compute lifting risks!")
        # for each predicted motion we retrieve NIOSH variables and compute risks
        for frame in range(len(motion_prediction_array)):
            human_jpos_data = motion_prediction_array[frame]
            # update s
            for i in range(len(human_jpos_data)):
                s.setVal(name_to_index[i], human_jpos_data[i])
            # update H_b
            # !!since prediction horizon is 1s, assume base pose not changing here
            H_b_np = nut.Quaternion2Mat()(base_orien_pos)
            H_b.fromHomogeneousTransform(idt.Matrix4x4_FromPython(H_b_np))

            # update huamn state
            dynCom.setRobotState(H_b, s, w_b, ds, gravity)

            # left hand w.r.t. world
            # H_link_lh_world = dynCom.getWorldTransform("LeftHand")
            H_link_lh_world = dynCom.getRelativeTransform("LeftFoot", "LeftHandCOM")
            H_link_lh_world_np = H_link_lh_world.asHomogeneousTransform().toNumPy()
            link_lh_world_pos = H_link_lh_world_np[:3, 3]
            links_lh_world.append(link_lh_world_pos)

            # right hand w.r.t. world
            # H_link_rh_world = dynCom.getWorldTransform("RightHand")
            H_link_rh_world = dynCom.getRelativeTransform("RightFoot", "RightHandCOM")
            H_link_rh_world_np = H_link_rh_world.asHomogeneousTransform().toNumPy()
            link_rh_world_pos = H_link_rh_world_np[:3, 3]
            # print("[ahhh]: ", link_rh_world_pos)
            links_rh_world.append(link_rh_world_pos)

            # left hand w.r.t. L5 joint
            H_link_lh_L5 = dynCom.getRelativeTransform("LeftFoot", "LeftHandCOM")
            H_link_lh_L5_np = H_link_lh_L5.asHomogeneousTransform().toNumPy()
            link_lh_L5_pos = H_link_lh_L5_np[:3, 3]
            links_lh_L5.append(link_lh_L5_pos)

            # right hand w.r.t. L5 joint
            H_link_rh_L5 = dynCom.getRelativeTransform("RightFoot", "RightHandCOM")
            H_link_rh_L5_np = H_link_rh_L5.asHomogeneousTransform().toNumPy()
            link_rh_L5_pos = H_link_rh_L5_np[:3, 3]
            links_rh_L5.append(link_rh_L5_pos)
    else: # links list of size 1
        print("[INFO] Use current sensor measurements to compute lifting risks!")
        # ===============================
        # update joints values
        for i in range(len(human_jPos_data)):
            s.setVal(name_to_index[i], human_jPos_data[i])
        
        # update H matrix
        H_b_np = nut.Quaternion2Mat()(base_orien_pos)
        H_b.fromHomogeneousTransform(idt.Matrix4x4_FromPython(H_b_np))

        # update the kinDynComputation object with H_b and s
        # the other values (w_b, ds, gravity) are fixed
        dynCom.setRobotState(H_b, s, w_b, ds, gravity)

        # ================================
        # get the desired link configuration w.r.t. world frame
        # for retrieving H, V and D
        # H_link_lh_world = dynCom.getWorldTransform("LeftHand")
        H_link_lh_world = dynCom.getRelativeTransform("LeftFoot", "LeftHandCOM")
        H_link_lh_world_np = H_link_lh_world.asHomogeneousTransform().toNumPy()
        link_lh_world_pos = H_link_lh_world_np[:3, 3]
        links_lh_world.append(link_lh_world_pos)

        # H_link_rh_world = dynCom.getWorldTransform("RightHand")
        H_link_rh_world = dynCom.getRelativeTransform("RightFoot", "RightHandCOM")
        H_link_rh_world_np = H_link_rh_world.asHomogeneousTransform().toNumPy()
        link_rh_world_pos = H_link_rh_world_np[:3, 3]
        links_rh_world.append(link_rh_world_pos)

        H_link_lh_L5 = dynCom.getRelativeTransform("LeftFoot", "LeftHandCOM")
        H_link_lh_L5_np = H_link_lh_L5.asHomogeneousTransform().toNumPy()
        link_lh_L5_pos = H_link_lh_L5_np[:3, 3]
        links_lh_L5.append(link_lh_L5_pos)

        H_link_rh_L5 = dynCom.getRelativeTransform("RightFoot", "RightHandCOM")
        H_link_rh_L5_np = H_link_rh_L5.asHomogeneousTransform().toNumPy()
        link_rh_L5_pos = H_link_rh_L5_np[:3, 3]
        links_rh_L5.append(link_rh_L5_pos)
    
    link_hands_L5_posx_list = []
    link_hands_world_posz_list = []

    # get the current recognized action label
    freeze_old_action_array, action_array_probs, action_index_now, action_label_now = nut.detectActionChange(freeze_old_action_array,
                                                                                                             action_array, 
                                                                                                             count,
                                                                                                             action_label_origin)
    # update geometry varibales of NIOSH equation
    if count == 0: # only for first running
        print("[INFO] First human action detection is: {} with probability {}".format(action_label_now, 
                                                                                      action_array_probs[action_index_now]))
        action_label_origin = action_label_now
        # 0 means x coordinate
        link_hands_L5_posx0 = nut.computePosition(links_lh_L5[0], links_rh_L5[0], 0)
        # 2 means z coordinate
        link_hands_world_posz0 = nut.computePosition(links_lh_world[0], links_rh_world[0], 2) 

        for i in range(len(links_lh_world)):
            link_hands_L5_posx = nut.computePosition(links_lh_L5[i], links_rh_L5[i], 0)
            link_hands_L5_posx_list.append(link_hands_L5_posx)
            link_hands_world_posz = nut.computePosition(links_lh_world[i], links_rh_world[i], 2)
            link_hands_world_posz_list.append(link_hands_world_posz)
  
        # link_hands_world_posz = link_hands_world_posz0
        # link_hands_L5_posx = link_hands_L5_posx0

    # in case detecting action change
    elif action_label_now is not action_label_origin:
        current_time = time.perf_counter()
        print("[INFO] Human action changes from: {} to: {} at time: {}".format((action_label_origin),
                                                                            (action_label_now),
                                                                            (current_time)))
        action_label_origin = action_label_now

        # update original hands x0 position w.r.t. L5 frame 
        link_hands_L5_posx0 = nut.computePosition(links_lh_L5[0], links_rh_L5[0], 0)
        # update original hands z0 position w.r.r. world frame
        link_hands_world_posz0 = nut.computePosition(links_lh_world[0], links_rh_world[0], 2)
        # update current hands z position w.r.t. world frame
        for i in range(len(links_lh_world)):
            link_hands_L5_posx = nut.computePosition(links_lh_L5[i], links_rh_L5[i], 0)
            link_hands_L5_posx_list.append(link_hands_L5_posx)
            link_hands_world_posz = nut.computePosition(links_lh_world[i], links_rh_world[i], 2)
            link_hands_world_posz_list.append(link_hands_world_posz)

        # link_hands_world_posz = link_hands_world_posz0
        # link_hands_L5_posx = link_hands_L5_posx0

    else: # in case detected action maintain unchanged
        print("[INFO] Human action remains the same as: {} with probability {}".format(action_label_now,
                                                                                       action_array_probs[action_index_now]))
        # update current hands z position w.r.t. world frame
        for i in range(len(links_lh_world)):
            link_hands_L5_posx = nut.computePosition(links_lh_L5[i], links_rh_L5[i], 0)
            link_hands_L5_posx_list.append(link_hands_L5_posx)
            link_hands_world_posz = nut.computePosition(links_lh_world[i], links_rh_world[i], 2)
            link_hands_world_posz_list.append(link_hands_world_posz)

        # link_hands_world_posz = nut.computePosition(link_lh_world_pos, link_rh_world_pos, 2)
        # link_hands_L5_posx = nut.computePosition(link_lh_L5_pos, link_rh_L5_pos, 0)

    # =============================
    ## prepare NIOSH parameters and multipliers
    NIOSH_paras_bottle = NIOSH_paras_port.prepare()
    NIOSH_paras_bottle.clear()

    # NIOSH_factors_bottle = NIOSH_factors_port.prepare()
    # NIOSH_factors_bottle.clear()

    # maybe we should read time logged in GMoE online test?
    log_time = time.perf_counter()
    NIOSH_paras_bottle.addFloat64(log_time)
    # NIOSH_factors_bottle.addFloat64(log_time)

    Hs, HMs, Vs, VMs, Ds, DMs, AMs, CMs, FMs, RWLs, RIs = ([] for i in range(11))
    # ==============================
    # hands_L5_xMax = link_hands_L5_posx_list[0]
    # hands_world_zMax = link_hands_world_posz_list[0]
    H_max = niosh.findH(0, link_hands_L5_posx_list[0])
    V_max = niosh.findV(0, link_hands_world_posz_list[0])
    for i in range(len(links_lh_world)):
        # ////////////////////////////////////////////
        # horizontal distance H & HM
        # use the original value of H to compute RWL
        #H = niosh.findH(0, link_hands_L5_posx0)
        # update H continuously
        # H = niosh.findH(0, link_hands_L5_posx0)
        H_now = niosh.findH(0, link_hands_L5_posx_list[i])
        if H_now >= H_max:
            H, H_max = H_now, H_now
        else:
            H = H_max
        # if link_hands_L5_posx_list[i] > hands_L5_xMax:
        #     H = niosh.findH(0, link_hands_L5_posx_list[i])
        #     hands_L5_xMax = link_hands_L5_posx_list[i]
        # else:
        #     H = niosh.findH(0, hands_L5_xMax)
        # H = niosh.findH(0, link_hands_L5_posx_list[i])
        Hs.append(H)
        #NIOSH_paras_bottle.addFloat64(H)
        HM = niosh.get_HM(H)
        HMs.append(HM)
        # NIOSH_factors_bottle.addFloat64(HM)

        # //////////////////////////////////////////////
        # vertical distance V & VM
        # V = niosh.findV(0, link_hands_world_posz0)
        V_now = niosh.findV(0, link_hands_world_posz_list[i])
        if V_now >= V_max:
            V, V_max = V_now, V_now
            hands_world_zMax = link_hands_world_posz_list[i]
        else:
            V = V_max
        # if link_hands_world_posz_list[i] >= hands_world_zMax:
        #     V = niosh.findV(0, link_hands_world_posz_list[i])
        #     hands_world_zMax = link_hands_world_posz_list[i]
        # else:
        #     V = niosh.findV(0, hands_world_zMax)
        # V = niosh.findV(0, link_hands_world_posz_list[i])
        Vs.append(V)
        # NIOSH_paras_bottle.addFloat64(V)
        VM = niosh.get_VM(V)
        VMs.append(VM)
        # NIOSH_factors_bottle.addFloat64(VM)

        # ////////////////////////////////////////////////
        # vertical traveling distance D & DM
        # D = niosh.findD(link_hands_world_posz0, link_hands_world_posz_list[i])
        D = niosh.findD(link_hands_world_posz0, hands_world_zMax)
        Ds.append(D)
        #NIOSH_paras_bottle.addFloat64(D)
        DM = niosh.get_DM(D)
        DMs.append(DM)
        # NIOSH_factors_bottle.addFloat64(DM)

        # ===============================
        # assymetry angle is as default zero 
        AM = niosh.get_AM()
        AMs.append(AM)
        # NIOSH_factors_bottle.addFloat64(AM)

        # coupling state default as 'Fair'
        CM = niosh.get_CM(V, 'Fair')
        CMs.append(CM)
        # NIOSH_factors_bottle.addFloat64(CM)

        # lifting frequecny as default 7 lifts/min
        FM = niosh.get_FM(V, duration=1)
        FMs.append(FM)
        # NIOSH_factors_bottle.addFloat64(FM)
        # ================================

        # prepare RWL 
        RWL_bottle = RWL_port.prepare()
        RWL_bottle.clear()
        RWL = niosh.get_RWL(HM, VM, DM, AM, CM, FM)
        RWLs.append(RWL)
        # RWL_bottle.addFloat64(RWL)
        # RWL_port.write()

        # prepare RI
        # RI_bottle = RI_port.prepare()
        # RI_bottle.clear()
        # consider only risk during rising
        if action_label_now in ncf.lifting_list:
            RI = niosh.get_RI(weight_load, RWL)
        # risk during standing and squatting are ignored
        else:
            RI = niosh.get_RI(0, RWL)
        RIs.append(RI)
        # RI_bottle.addFloat64(RIs)
        # RI_bottle.addList(RIs)
        
        # write to yarp ports
        # NIOSH_paras_port.write()
        # NIOSH_factors_port.write()
        # RWL_port.write()
        # RI_port.write()
    
    NIOSH_paras_bottle.addFloat64(Hs[0])
    NIOSH_paras_bottle.addFloat64(Vs[0])
    NIOSH_paras_bottle.addFloat64(Ds[0])
    NIOSH_paras_port.write()
    # RWL_bottle = RWL_port.prepare()
    # RWL_bottle.clear()
    RI_bottle = RI_port.prepare()
    RI_bottle.clear()
    command_ifeel_bottle = command_ifeel_port.prepare()
    command_ifeel_bottle.clear()
    ergocub_demo_bottle = ergocub_demo_port.prepare()
    ergocub_demo_bottle.clear()

    for i in range(len(RWLs)):
        # RWL_bottle.addFloat64(RWLs[i])
        RI_bottle.addFloat64(RIs[i])
    # RWL_port.write()
    RI_port.write()

    # create a bottle to hold the ActuatorInfo
    # More info check below: 
    # https://github.com/robotology/wearables/blob/master/msgs/thrift/WearableActuators.thrift
    #command_info_bottle = command_ifeel_bottle.addList()
    # specify the ifeel node name: just change the number in "Node#x"
    # More info check below:
    # https://github.com/ami-iit/component_ifeel/blob/master/element_wearable_sw/iFeelSuit/src/iFeelSuit.cpp#L327
    actuator_name_prefix = "iFeelSuit::haptic::Node#"
    actuator_name = actuator_name_prefix + str(4)
    #command_info_bottle.addString(actuator_name)
    # actuator type 0 means haptic
    #command_info_bottle.addInt32(0)
    command_ifeel_bottle.addString(actuator_name)
    command_ifeel_bottle.addInt32(0)
    # check the values in lifting index list
    if all(risk < 1.0 for risk in RIs):
        # ifeel node actuator vibration amplitude
        command_ifeel_bottle.addFloat64(0.0)
        #ergocub_demo_bottle.addString("Safe working condition.")
    elif any(2.0 > risk > 1.0 for risk in RIs):
        command_ifeel_bottle.addFloat64(0.3)
        ergocub_demo_bottle.addString("Medium")
    elif any(3.0 > risk > 2.0 for risk in RIs):
        command_ifeel_bottle.addFloat64(0.6)
        ergocub_demo_bottle.addString("High")
    elif any(risk > 3.0 for risk in RIs):
        command_ifeel_bottle.addFloat64(1.0)
        ergocub_demo_bottle.addString("DANGEROUS!")

    command_ifeel_bottle.addFloat64(0)
    command_ifeel_port.write()

    ergocub_demo_port.write()

    # update iteration
    count += 1




