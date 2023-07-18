# ================================================================================================
#
# This function mainly visualizes the significance of lifting risk via human model. 
#
# ref: ami-iit/element_ergonomy-control/ergonomy-optimization/adam-optimization/visualizer.py
# ================================================================================================

#from msilib import type_binary
from unittest.util import _count_diff_all_purpose
import yarp
import idyntree.bindings as iDynTree
import numpy as np
import casadi as cs
import time 

class humanModelVisualizer:
    # initialize the huamn model visualizer
    def __init__(self, model1_name, model2_name, base_link="RightFoot", 
                 color_palette = "meshcat", force_scale_factor = 0.001):
        self.subject_name = model1_name
        self.prediction_name = model2_name
        self.base_link = base_link
        
        self.urdf_path = "human_model_urdf/humanSubject01_66dof.urdf"
        self.joint_list = ["jT9T8_rotx",
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
              "jLeftKnee_rotz"
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
        #self.H_B = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
        #                      [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
        self.f_c_subject = None
        self.f_c_prediction = None
    
        self.idyntree_visualizer = iDynTree.Visualizer()
        super().__init__
        visualizer_options = iDynTree.VisualizerOptions()
        self.idyntree_visualizer.init(visualizer_options)
        self.idyntree_visualizer.setColorPalette(color_palette)
        self.force_scale_factor = force_scale_factor

        #self.human_joint_pos_port = yarp.BufferedPortBottle()
    # initialize yarp network
    #def yarpConf(self):
    #    yarp.Network.init()
    #    if not yarp.Network.checkNetwork():
    #        print("[ERROR] Unable to open a YARP Network.")
    #    else:
    #        print("[INFO] Open a YARP Network happily.")
        
        # get joint configuration
    #     human_joint_pos_port = yarp.BufferedPortBottle()
    #    self.human_joint_pos_port.open("/risk_prediction/jointPositions:i")
    #    jPos_is_connected = yarp.Network.connect("/risk_prediction/jointPositions:o", "/risk_prediction/jointPositions:i")
    #    print("Human joint positions port is connected: {}".format(jPos_is_connected))

    def load_model(self, model1_color = None, model2_color = None):
        model_subject_Loader_init = iDynTree.ModelLoader()
        model_prediction_Loader_init = iDynTree.ModelLoader()

        model_subject_Loader_init.loadReducedModelFromFile(self.urdf_path, self.joint_list, "urdf")
        model_prediction_Loader_init.loadReducedModelFromFile(self.urdf_path, self.joint_list, "urdf")
        
        # workaround: force the default base frame to be the desired one
        desired_linkIndex_sub = model_subject_Loader_init.model().getLinkIndex(self.base_link)
        model_subject_Loader_init.model().setDefaultBaseLink(desired_linkIndex_sub)
        desired_linkIndex_pred= model_prediction_Loader_init.model().getLinkIndex(self.base_link)
        model_prediction_Loader_init.model().setDefaultBaseLink(desired_linkIndex_pred)

        # check the deafult base frame
        linkIndex = model_subject_Loader_init.model().getDefaultBaseLink()
        linkName = model_subject_Loader_init.model().getLinkName(linkIndex)
        print("[INFO] Base frame is: {}".format(linkName))

        self.idyntree_visualizer.addModel(model_subject_Loader_init.model(), self.subject_name)
        self.idyntree_visualizer.addModel(model_prediction_Loader_init.model(), self.prediction_name)

        # when requiring a different model color
        if not model1_color is None:
            self.idyntree_visualizer.modelViz(self.subject_name).setModelColor(iDynTree.ColorViz(iDynTree.Vector4_FromPython(model1_color)))
        if not model2_color is None:
            self.idyntree_visualizer.modelViz(self.prediction_name).setModelColor(iDynTree.ColorViz(iDynTree.Vector4_FromPython(model2_color)))
    # set model configuration
    # given s, and H_B representing joint configuration and base transform as numpy objects
    def update_model(self, s1, s2, H_B):
        s_idyntree_opti_sub = iDynTree.VectorDynSize.FromPython(s1)
        #T_b_opti_sub = iDynTree.Transform()
        #_b_opti_sub.fromHomogeneousTransform(iDynTree.Matrix4x4(H_B))

        s_idyntree_opti_pred = iDynTree.VectorDynSize.FromPython(s2)
        #T_b_opti_pred = iDynTree.Transform()
        #T_b_opti_pred.fromHomogeneousTransform(iDynTree.Matrix4x4(H_B))
        
        T_b_opti = iDynTree.Transform()
        T_b_opti.fromHomogeneousTransform(iDynTree.Matrix4x4(H_B))
        # update the model configuration using 'setPositions'
        self.idyntree_visualizer.modelViz(self.subject_name).setPositions(T_b_opti, s_idyntree_opti_sub)
        self.idyntree_visualizer.modelViz(self.prediction_name).setPositions(T_b_opti, s_idyntree_opti_pred)

    # run the visualizer while reading data from yarp ports
    def run(self):
        # self.idyntree_visualizer.camera().animator().enableMouseControl()
        #self.idyntree_visualizer.run()
        self.idyntree_visualizer.draw()

        #while(self.idyntree_visualizer.run()):
            #human_jPositions = human_joint_pos_port.read()
            # update the model configration using 'setPositions'
            #self.idyntree_visualizer.draw()

# convert quaternion to transform matrix
def Quaternion2Mat():
    H = cs.SX.eye(4)
    q = cs.SX.sym("q", 7)

    R = cs.SX.eye(3) + 2*q[0]*cs.skew(q[1:4]) + 2*cs.mpower(cs.skew(q[1:4]), 2)

    H[:3, :3] = R
    H[:3, 3] = q[4:7]
    H = cs.Function("H", [q], [H])

    return H

def current_milli_time():
    return round(time.time() * 1000)


# initialize the yarp network
yarp.Network.init()
if not yarp.Network.checkNetwork():
    print("[ERROR] Unable to open a YARP Network.")
else:
    print("[INFO] Open a YARP Network happily.")

# get human subject joint configuration
human_joint_pos_port = yarp.BufferedPortBottle()
human_joint_pos_port.open("/risk_prediction/jointPositions:i")
jPos_is_connected = yarp.Network.connect("/risk_prediction/jointPositions:o", 
                                         "/risk_prediction/jointPositions:i")
#print("Human joint positions port is connected: {}".format(jPos_is_connected))

# get human prediction joint configuration
prediction_jPos_port = yarp.BufferedPortBottle()
prediction_jPos_port.open("/risk_prediction/prediction_jPos:i")
prediction_is_connected = yarp.Network.connect("/test_moe/motionPrediction:o",
                                               "/risk_prediction/prediction_jPos:i")
#print("Human prediction joint positions is connected: {}".format(prediction_is_connected))

# get base positions and orientations
base_pos_orien_port = yarp.BufferedPortBottle()
base_pos_orien_port.open("/risk_prediction/basePosOrien:i")
basePosOrien_is_connected = yarp.Network.connect("/humanDataAcquisition/basePose:o", 
                                                 "/risk_prediction/basePosOrien:i")
#print("Human base positions and orientations port is connected: {}".format(basePosOrien_is_connected))

# create human subject visualizer
#humanSubject = humanModelVisualizer(model_name="humanSubject")
#humanSubject.load_model(model_color=(0.2 , 0.2, 0.2, 0.9))

# create human prediction visualizer
#humanPrediction = humanModelVisualizer(model_name="humanPrediction")
#humanPrediction.load_model(model_color=(1.0 , 0.2, 0.2, 0.5))

humanModel = humanModelVisualizer(model1_name="humanSubject", model2_name="humanPrediction")
humanModel.load_model(model1_color=(0.2 , 0.2, 0.2, 0.9), model2_color=(1.0 , 0.2, 0.2, 0.5))
#solver = cs.Opti
joint_list_sub = ["jT9T8_rotx",
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
              "jLeftKnee_rotz"
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

joint_list_full = ["jL5S1_rotx" , "jRightHip_rotx" , "jLeftHip_rotx" , "jLeftHip_roty" , "jLeftHip_rotz" , "jLeftKnee_rotx" , "jLeftKnee_roty" ,
                "jLeftKnee_rotz" , "jLeftAnkle_rotx" , "jLeftAnkle_roty" , "jLeftAnkle_rotz" , "jLeftBallFoot_rotx" , "jLeftBallFoot_roty" ,
                "jLeftBallFoot_rotz" , "jRightHip_roty" , "jRightHip_rotz" , "jRightKnee_rotx" , "jRightKnee_roty" , "jRightKnee_rotz" ,
                "jRightAnkle_rotx" , "jRightAnkle_roty" , "jRightAnkle_rotz" , "jRightBallFoot_rotx" , "jRightBallFoot_roty" , "jRightBallFoot_rotz" ,
                "jL5S1_roty" , "jL5S1_rotz" , "jL4L3_rotx" , "jL4L3_roty" , "jL4L3_rotz" , "jL1T12_rotx" , "jL1T12_roty" , "jL1T12_rotz" ,
                "jT9T8_rotx" , "jT9T8_roty" , "jT9T8_rotz" , "jLeftC7Shoulder_rotx" , "jT1C7_rotx" , "jRightC7Shoulder_rotx" , "jRightC7Shoulder_roty" ,
                "jRightC7Shoulder_rotz" , "jRightShoulder_rotx" , "jRightShoulder_roty" , "jRightShoulder_rotz" , "jRightElbow_rotx" , "jRightElbow_roty" ,
                "jRightElbow_rotz" , "jRightWrist_rotx" , "jRightWrist_roty" , "jRightWrist_rotz" , "jT1C7_roty" , "jT1C7_rotz" , "jC1Head_rotx" ,
                "jC1Head_roty" , "jC1Head_rotz" , "jLeftC7Shoulder_roty" , "jLeftC7Shoulder_rotz" , "jLeftShoulder_rotx" , "jLeftShoulder_roty" ,
                "jLeftShoulder_rotz" , "jLeftElbow_rotx" , "jLeftElbow_roty" , "jLeftElbow_rotz" , "jLeftWrist_rotx" , "jLeftWrist_roty" ,
                "jLeftWrist_rotz"]

humanModel.idyntree_visualizer.camera().animator().enableMouseControl()

while(humanModel.idyntree_visualizer.run()):
    time1 = current_milli_time()
    human_jPositions = human_joint_pos_port.read()

    time21 = current_milli_time()
    print ("show time21: ", time21-time1)

    #human_joint_pos_port.setStrict(True)
    #human_joint_pos_port.setTargetPeriod(0.04)
    prediction_jPos = prediction_jPos_port.read()

    time22 = current_milli_time()
    print ("show time22: ", time22-time21)

    #prediction_jPos_port.setStrict(True)
    #prediction_jPos_port.setTargetPeriod(0.04)
    base_pos_orien = base_pos_orien_port.read()

    time23 = current_milli_time()
    print ("show time23: ", time23-time22)

    #base_pos_orien_port.setStrict(True)
    #base_pos_orien_port.setTargetPeriod(0.04)
    time2 = current_milli_time()
    print ("show time2: ", time2-time1)
    
    if human_jPositions is not None and base_pos_orien is not None:
        #print("[INFO] Read human joint configuration successfully!")
        #print("[INFO] Read human joint prediction configuration successfully!")
        #print("[INFO] Read base positions and orientations successfully!")

        human_jPos_data = []
        human_prediction_data = []
        base_pos = []
        base_orien = []
        base_orientation_pos = []

        for i in range(human_jPositions.size()):
            human_jPos_data.append(human_jPositions.get(i).asFloat64())

        for i in range(prediction_jPos.size()):
            human_prediction_data.append(prediction_jPos.get(i).asFloat64())
        
        for i in range(base_pos_orien.size()):
            if i < 3:
                base_pos.append(base_pos_orien.get(i).asFloat64())
            else:
                base_orien.append(base_pos_orien.get(i).asFloat64())
        base_orientation_pos = np.concatenate((base_orien, base_pos))
        
        #print("[INFO] Streaming human joint positions: ", human_jPos_data)
        #print(type(human_jPos_data))
        #print("[INFO] Streaming human joint prediction positions: ", human_prediction_data)
        #print(type(human_prediction_data))
        #print("[INFO] Streaming base positions and orientations: ", base_orientation_pos)

        # update subject joint configuration s
        #conf_subject = [0]*66
        #conf_prediction = [0]*66
        #for i in range(len(joint_list_sub)):
        #    if joint_list_sub[i] in joint_list_full:
        #        # find the index of the joint in full joint list
        #        jointIndex = joint_list_full.index(joint_list_sub[i])
        #        # update the joint configuration list for both models
        #        conf_subject[jointIndex] = human_jPos_data[i]
        #        conf_prediction[jointIndex] = human_prediction_data[i]

        conf_subject = human_jPos_data
        #for i in range(len(joint_list_full)):
        #    if joint_list_full[i] not in joint_list_sub:
        #        conf_subject[i] = 0

        #print("[INFO] conf_subject list: ", conf_subject)
        # update human prediction joint configuration
        #conf_prediction = [0] * len(human_jPos_data)
        #print("[INFO] original conf_prediction: ", conf_prediction)
        #for i in range(len(human_prediction_data)):
        #    conf_prediction[joint_prediction_index_List[i]] = human_prediction_data[i]
        #print("[INFO] now conf_prediction: ", conf_prediction)
    
        conf_prediction = human_prediction_data
        #print("[INFO] conf_prediction list: ", conf_prediction)

        # update base transform matrix H_B
        H_B = Quaternion2Mat()(base_orientation_pos)
        #H_B = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
        #                 [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])

        time3 = current_milli_time()

        print ("show time3: ", time3-time2)

        #visualizer = humanModelVisualizer()
        #visualizer.load_model()
        humanModel.update_model(conf_subject, conf_prediction, H_B)
        humanModel.run()

        time4 = current_milli_time()
        print ("show time4: ", time4-time3)
        
        #humanPrediction.update_model(conf_prediction, H_B)
        #humanPrediction.run()
    else:
        print("[ERROR] Human joint positions are none: ", human_jPositions)

#human_jPositions = human_joint_pos_port.read()
#yarp.delay(5)
#print("Look at this: ", human_jPositions)

#if human_jPositions is not None:
#    print("Read data successfully!")
#    human_jPos_data = []
#    for i in range(human_jPositions.size()):
#        human_jPos_data.append(human_jPositions.get(i).asFloat64())
#    
#    print("human joint positions: ", human_jPos_data)
#    conf = human_jPos_data

    # initialize the visualizer
#    visualizer = humanModelVisualizer(model_name)

    # update the model
#    visualizer.load_model()
#    visualizer.update_model(conf)
#    visualizer.run()
