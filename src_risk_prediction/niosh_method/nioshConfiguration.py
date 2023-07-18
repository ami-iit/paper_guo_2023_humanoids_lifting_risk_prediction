# ====================#
#                     #
# NIOSH CONFIGURATION #
#                     #
# ====================#

#action_labels = ["none", "standing", "stooping", "bending", "straightening", "rising", "placing",
#                 "fetching", "stoop-lowering", "bend-lowering", "stoop-back", "bend-back"]
action_labels = ["rising", "squatting", "standing"]
#lifting_list = ["straightening", "rising", "placing", "stoop-lowering", "bend-lowering"]
lifting_list = ["rising"]
#non_lifting_list = ["none", "standing", "stooping", "bending", "fetching", "stoop-back", "bend-back"]
non_lifting_list = ["standing", "squatting"]

plot_results = True
use_motion_predictions = True
use_threshold = True
use_maxProb = False

weight = 10

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