'''
To evaluate the performance of GMoE regarding:
# Action recognition:
- confusion matrix (multi-class classification)
- should we create a metric that considers the fastness of detecting action transition??

# Motion prediction:
- Choose multiple joints, plot their ground truth, predictions, and error region
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics

from Utilities import load_model_from_file
from DatasetUtility import current_milli_time

# normalize 12-dim contact forces with human weight
def normalize_human_weight(data, user_mass, gravity):
    print('data type is: ', type(data))
    # convert to array
    data_as_array = data.to_numpy()
    print('data array shape is: ', data_as_array.shape)

    user_weight = user_mass * gravity
    for i in range(data.shape[0]):
        for j in range(62, len(data_as_array[i])):
            data_as_array[i][j] = data_as_array[i][j] / user_weight

    data_normalized = pd.DataFrame(data_as_array)
    print('normalized data is: ', data_normalized)

    return data_normalized

# TO DO: to retrieve predicted contact wrenches
""" def denormalize_motion_wrench_predictions(pred):
    return  """

if __name__ == "__main__":
    # parameters setting
    history_steps = 10
    output_steps = 50
    window_size = history_steps + output_steps
    stride = 1 # this is to obtain action prediction for each frame
    user_mass = 78
    gravity = 9.81
    num_input_features = 74
    fontsize = 18

    motion_pred_slices = slice(0, 31)
    wrench_pred_slices = slice(31, 43)

    desired_action_idx = 0
    desired_motion_idx = [0, 19, 49]

    evaluate_action_recognition = False
    evaluate_motion_prediction = True

    action_labels = ["rising", "squatting", "standing"]
    # note here the idx considers also 'time' column
    # in predictions, should be idx-1
    jLeftKnee_roty_idx = 17
    jRightElbow_roty_idx = 7

    # prepare the trained GMoE model
    model_name = 'model_MoE_Best'
    model_path = 'NN_models/2023-02-23 14:30:03'
    model = load_model_from_file(file_path=model_path, file_name=model_name)

    # prepare the offline data
    data_path = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/01_cheng_labeled.txt' 
    data = pd.read_csv(data_path, sep=' ')
    shape_ = data.shape

    # prepare ground truth for action prediction
    # get the column of 'label' as action ground truth for each frame
    # i.e. ['rising', 'standing', ..., 'squatting', ...]
    action_gt = data.iloc[history_steps:, -1].to_numpy()

    # prepare ground truth for motion predictions
    # two joints: left knee and right elbow, around y-axis rotation
    motion_gt_jLeftKnee_roty = data.iloc[:, jLeftKnee_roty_idx].to_numpy()
    motion_gt_jRightElbow_roty = data.iloc[:, jRightElbow_roty_idx].to_numpy()

    # remove the columns of 'time' and 'label'
    pop_list = ['time', 'label']
    for pop_name in pop_list:
        data.pop(pop_name)
    print('pop data is: ', data)

    # normalize contact forces with human weight
    data_normalized = normalize_human_weight(data, user_mass, gravity)

    if evaluate_action_recognition:
        action_preds = []
        action_probs = []
        # test on the whole dataset
        tik_total = current_milli_time()
        for idx in range(0, data_normalized.shape[0]-history_steps, stride):
            print('start evaluating action recognition, please wait...')
            # every time get 10 frames as inputs
            inputs = data_normalized.iloc[idx:idx+history_steps, :].to_numpy()
            inputs = np.reshape(inputs, (1, history_steps, num_input_features))

            # do the predictions
            predictions = model(inputs, training=False)

            # original predictions[0] is eager tensor
            # retrieve action predictions, shape (1, 50, 3)
            action_predictions_all = np.float64(np.array(predictions[0]))

            # reshape to (50, 3)
            if np.size(action_predictions_all.shape) > 2:
                action_predictions_all = np.reshape(action_predictions_all,
                                                    (action_predictions_all.shape[1], action_predictions_all.shape[2]))

            # collect only the action prediction at first time step
            desired_action_pred = action_predictions_all[desired_action_idx, :]
            print('current frame is {}, desired action prediction is {} '.format(idx, desired_action_pred))

            # get the action label with max probability
            max_idx = np.argmax(desired_action_pred)
            action_ = action_labels[max_idx]
            action_prob_ = desired_action_pred[max_idx]

            # append predicted action label to list 
            action_preds.append(action_)
            action_probs.append(action_prob_)

        tok_total = current_milli_time()
        print('Action recognition evaluation finished, total time cost {}'.format((tok_total-tik_total)))

        print('action preds shape is: ', len(action_preds))
        print('action gt shape is: ', len(action_gt))

        # compute the confusion matrix of action predictions
        confusion_matrix = metrics.confusion_matrix(action_gt, action_preds, labels=["rising", "squatting", "standing"])

        # plot the results
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["rising", "squatting", "standing"])

        cm_display.plot()
        plt.show()


    if evaluate_motion_prediction:
        motion_gt_jLeftKnee = []
        motion_gt_jRightElbow = []

        motion_preds_jLeftKnee_roty_idx0 = []
        motion_preds_jLeftKnee_roty_idx19 = []
        motion_preds_jLeftKnee_roty_idx49 = []

        motion_preds_jRightElbow_roty_idx0 = []
        motion_preds_jRightElbow_roty_idx19 = []
        motion_preds_jRightElbow_roty_idx49 = []

        tik_total = current_milli_time()
        for idx in range(0, data_normalized.shape[0]-history_steps-output_steps-9700, stride):
            print('start evaluating motion predictions, please wait...')
            # get inputs
            inputs = data_normalized.iloc[idx:idx+history_steps, :].to_numpy()
            inputs = np.reshape(inputs, (1, history_steps, num_input_features))

            # do the predictions
            predictions = model(inputs, training=False)

            #retrieve motion and wrench predictions for all future time steps, i.e. 50 frames
            motion_wrench_predictions_all = np.float64(np.array(predictions[1]))
            # reshape to (50, 43)
            if np.size(motion_wrench_predictions_all.shape) > 2:
                motion_wrench_predictions_all = np.reshape(motion_wrench_predictions_all,
                                                           (motion_wrench_predictions_all.shape[1], 
                                                            motion_wrench_predictions_all.shape[2]))

            motion_predictions_all = motion_wrench_predictions_all[:, motion_pred_slices]
            
            # obtain predictions of left knee roty at time steps 0, 19 and 49
            motion_preds_jLeftKnee_idx0 = motion_predictions_all[desired_motion_idx[0], jLeftKnee_roty_idx-1]
            motion_preds_jLeftKnee_idx19 = motion_predictions_all[desired_motion_idx[1], jLeftKnee_roty_idx-1]
            print('current frame is {}, motion_preds_jLeftKnee_idx19 is {}'.format(idx, motion_preds_jLeftKnee_idx19))
            motion_preds_jLeftKnee_idx49 = motion_predictions_all[desired_motion_idx[2], jLeftKnee_roty_idx-1]
            # convert to degree and add to lists
            motion_preds_jLeftKnee_roty_idx0.append(np.degrees(motion_preds_jLeftKnee_idx0))
            motion_preds_jLeftKnee_roty_idx19.append(np.degrees(motion_preds_jLeftKnee_idx19))
            motion_preds_jLeftKnee_roty_idx49.append(np.degrees(motion_preds_jLeftKnee_idx49))
           
            # obtain predictions of right elbow roty at time steps 0, 19 and 49
            motion_preds_jRightElbow_idx0 = motion_predictions_all[desired_motion_idx[0], jRightElbow_roty_idx-1]
            motion_preds_jRightElbow_idx19 = motion_predictions_all[desired_motion_idx[1], jRightElbow_roty_idx-1]
            motion_preds_jRightElbow_idx49 = motion_predictions_all[desired_motion_idx[2], jRightElbow_roty_idx-1]
            # convert to degrees and add to lists
            motion_preds_jRightElbow_roty_idx0.append(np.degrees(motion_preds_jRightElbow_idx0))
            motion_preds_jRightElbow_roty_idx19.append(np.degrees(motion_preds_jRightElbow_idx19))
            motion_preds_jRightElbow_roty_idx49.append(np.degrees(motion_preds_jRightElbow_idx49))

            # prepare motion ground truth
            motion_gt_jLeftKnee.append(np.degrees(motion_gt_jLeftKnee_roty[idx+history_steps]))
            motion_gt_jRightElbow.append(np.degrees(motion_gt_jRightElbow_roty[idx+history_steps]))
            
            # for now we are not using wrench predictions, so just ignore them
            #wrench_predictions_all = motion_wrench_predictions_all[:, wrench_pred_slices]

        tok_total = current_milli_time()
        print('Motion prediction evaluation finished, total time cost {}'.format((tok_total-tik_total)))

        print('left knee action preds shape is: ', len(motion_preds_jLeftKnee_roty_idx19))
        print('left knee motion gt shape is: ', len(motion_gt_jLeftKnee))

        if len(motion_gt_jLeftKnee) != len(motion_preds_jLeftKnee_roty_idx19):
            print('manually aligh the size of ground truth and predictions')

        # prepare the plots
        x_ = np.arange(0, len(motion_gt_jLeftKnee))

        plt.figure(1)

        plt.subplot(211)
        plt.plot(x_, motion_gt_jLeftKnee, 'k', label='ground truth')
        plt.plot(x_, motion_preds_jLeftKnee_roty_idx0, color="#0071C2", label='prediction at time 0')
        plt.plot(x_, motion_preds_jLeftKnee_roty_idx19, color="#D75615", label='prediction at time 19')
        plt.plot(x_, motion_preds_jLeftKnee_roty_idx49, color="#EDB11A", label='prediction at time 49')
        #plt.xlabel("time step", fontsize=18)
        plt.ylabel("y-axis rotation angle [degree]", fontsize=19)
        plt.legend(loc="upper right", fontsize=18)
        plt.title("Joint left knee y-axis rotation angle.", fontsize=19)

        plt.subplot(212)
        plt.plot(x_, motion_gt_jRightElbow, 'k', label='ground truth')
        plt.plot(x_, motion_preds_jRightElbow_roty_idx0, color="#0071C2", label='prediction at time 0')
        plt.plot(x_, motion_preds_jRightElbow_roty_idx19, color="#D75615", label='prediction at time 19')
        plt.plot(x_, motion_preds_jRightElbow_roty_idx49, color="#EDB11A", label='prediction at time 49')
        plt.xlabel("time step", fontsize=19)
        plt.ylabel("y-axis rotation angle [degree]", fontsize=19)
        plt.legend(loc="upper right", fontsize=18)
        plt.title("Joint right elbow y-axis rotation angle.", fontsize=19)

        plt.show()







