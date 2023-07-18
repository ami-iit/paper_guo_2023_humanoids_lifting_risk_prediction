# ================#
#                 #
# NIOSH UTILITIES #
#                 #
# ================#
import casadi as cs
import time
import numpy as np
import nioshConfiguration as ncf

class nioshOnlineInference:
    def __init__(self):
        # define constant metrics
        self.LC = 23 # kg
        self.FREQUENCIES = np.array([0.2, 0.5, 1, 2, 3, 4,
                                     5, 6, 7, 8, 9, 10, 11,
                                     12, 13, 14, 15, 15.5]) # min 0.2lifts/min, max 15lifts/min
        self.DURATIONS = [1, 2, 8] # 1:lifting<1h, 2:lifitng 1~2h, 8:lifitng  2~8h
        self.CUTOFFS = {1: 14, 2: 12, 8: 10}
        self.FM_TABLE = {1: [1.00, 0.97, 0.94, 0.91, 0.88, 0.84,
                             0.80, 0.75, 0.70, 0.60, 0.52, 0.45,
                             0.41, 0.37, 0.34, 0.31, 0.28, 0.00],
                         2: [0.95, 0.92, 0.88, 0.84, 0.79, 0.72,
                             0.60, 0.50, 0.42, 0.35, 0.30, 0.26,
                             0.23, 0.21, 0.00, 0.00, 0.00, 0.00],
                         8: [0.85, 0.81, 0.75, 0.65, 0.55, 0.45,
                             0.35, 0.27, 0.22, 0.18, 0.15, 0.13,
                             0.00, 0.00, 0.00, 0.00, 0.00, 0.00]}
        self.COUPLINGS = {"Good": [1.0, 1.0], "Fair": [0.95, 1.00], "Poor": [0.9, 0.9]}
        
        # define NIOSH equation variables
        self.A = 0.0 # Angle of Asymmerty
        self.F = 7.0 # lifts per minute

    def duration_multiplier(self, V, duration, fmIndex):
        if V >= 30: # cm
            return 1
        else:
            return fmIndex <= self.CUTOFFS[duration]
    
    # find horizontal distance H
    def findH(self, jL5S1_x0, jHand_x0):
        return np.abs(jHand_x0 - jL5S1_x0) * 100
        
    # compute horizontal multiplier factor
    def get_HM(self, H):
        if H <= 63: # cm
            return 25.0 / max(25.0, H)
        else:
            return 0
    
    # find vertical distance V
    def findV(self, ground, jHand_z0):
        return np.abs(jHand_z0 - ground) * 100

    # compute vertical multiplier factor
    def get_VM(self, V):
        if V <= 175: # cm
            return 1 - (0.003 * np.abs(max(0, min(175, V)) - 75))
        else:
            return 0

    # find vertical moveing distance D
    def findD(self, jHand_z0, jHand_zt):
        return np.abs(jHand_zt - jHand_z0) * 100

    # compute distance multiplier factor
    def get_DM(self, D):
        return 0.82 + 4.5 / min(max(25.0, D), 175)
    
    # compute frequency multiplier factor
    def get_FM(self, V, duration):
        ind = np.argmin(np.abs(self.F - self.FREQUENCIES))
        multiplier = self.duration_multiplier(V, duration, ind)
        return self.FM_TABLE[duration][ind] * multiplier

    # compute asymmetric multiplier factor
    def get_AM(self): # degrees
        if self.A <= 135:
            return 1 - 0.0032 * self.A
        else:
            return 0

    # compute coupling multiplier factor
    def get_CM(self, V, coupling):
        if V >= 30:
            return self.COUPLINGS[coupling][1]
        else:
            return self.COUPLINGS[coupling][0]

    # compute RWL
    def get_RWL(self, HM, VM, DM, AM, CM, FM):
        return self.LC * HM * VM * DM * AM * CM * FM

    # compute RI 
    def get_RI(self, load_weight, RWL):
        if RWL == 0:
            return 10
        else:
            return load_weight / RWL

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

def computePosition(lvalue_vec, rvalue_vec, direction):
    left_pos = lvalue_vec[direction]
    right_pos = rvalue_vec[direction]

    return (left_pos + right_pos) / 2 


def detectActionChange(freeze_old_action_array, action_array, count, action_label_origin):
    # get the current recognized action label
    if ncf.use_threshold: # use threshold method to detect action change
        if count == 0:
            action_array_probs = action_array[0, :]
            action_index_now = action_array_probs.tolist().index(max(action_array_probs))
            action_label_now = ncf.action_labels[action_index_now]

            freeze_old_action_array = action_array
        else:
            # read action probability at current time
            action_array_probs = action_array[0, :]
            action_array_probs_old = freeze_old_action_array[0, :]

            # check if the probability of currently recognized action decreases
            recognized_action_idx = ncf.action_labels.index(action_label_origin)
            old_recog_action_prob = action_array_probs_old[recognized_action_idx]
            now_recog_action_prob = action_array_probs[recognized_action_idx]

            if now_recog_action_prob >= old_recog_action_prob: # the probability of current recognized action still grows
                # no need to change original action label
                action_index_now = recognized_action_idx
                action_label_now = action_label_origin
                # update last action array
                freeze_old_action_array = action_array
            elif old_recog_action_prob - now_recog_action_prob < 0.1:
                action_index_now = recognized_action_idx
                action_label_now = action_label_origin 
            else: # the probability of current recognized action falls
                # need to change original action label
                # detect which action probbility increases
                for i in range(len(action_array_probs)):
                    if (ncf.action_labels[i] != action_label_origin) and (action_array_probs[i] > action_array_probs_old[i]):
                        action_index_now = i
                        action_label_now = ncf.action_labels[i]
            # update last 
    
    elif ncf.use_maxProb: # use max probability to detect action change
        action_array_probs = action_array[0, :]
        action_index_now = action_array_probs.tolist().index(max(action_array_probs))
        action_label_now = ncf.action_labels[action_index_now]

    return freeze_old_action_array, action_array_probs, action_index_now, action_label_now
