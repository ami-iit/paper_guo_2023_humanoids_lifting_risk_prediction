# ===========================================================================
#
# This function mainly visualizes the significance of evaluated risk levels 
# based on biomechanical human joint limits via real-time animation.
#
# by Cheng Guo
# ============================================================================
import time
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.pylab import *
import yarp

def current_milli_time():
    return round(time.time() * 1000)

# initialize yarp network
yarp.Network.init()
if not yarp.Network.checkNetwork():
    print("[ERROR] Unable to open a YARP Network.")
else:
    print("[INFO] Open a YARP Network happily.")

# initialize yarp porte for receiving NIOSH results
read_RI_port = yarp.BufferedPortBottle()
read_RI_port.open("/risk_prediction/readRIValue:i")
RI_is_connected = yarp.Network.connect("/risk_prediction/risk_index:o",
                                       "/risk_prediction/readRIValue:i")
print("[INFO] Read RI value port is connected: {}".format(RI_is_connected))

# read_NIOSH_factors_port = yarp.BufferedPortBottle()
# read_NIOSH_factors_port.open("/risk_prediction/readNIOSHFactors:i")
# NIOSH_factos_is_connected = yarp.Network.connect("/risk_prediction/niosh_factors:o",
#                                                  "/risk_prediction/readNIOSHFactors:i")
# print("[INFO] Read NIOSH multipliers port is connected: {}".format(NIOSH_factos_is_connected))
yarp.delay(0.1)

class nioshResultsAnimation:
    def __init__(self):
        font = {'size': 15}
        mpl.rc('font', **font)
        
        self.xmin = 0.0
        self.xmax = 10.0
        self.plot_front_time = 1.2

        self.fig2 = figure(num=2, figsize=(8, 3.5))
        self.ax02 = self.fig2.subplots()
        self.ax02.set_title("Lifting index predictions during lifting task", fontsize=16)
        self.ax02.set_ylim(-0.5, 2.5)
        self.ax02.set_xlim(self.xmin, self.xmax)

        self.t = np.zeros(0)
        self.t0 = current_milli_time() / 1000.0
        self.t_prediction = np.zeros(0)

        self.RI_values = np.zeros(0)
        self.RI_predictions = np.zeros(0)
        # self.HM_list = np.zeros(0)
        # self.VM_list = np.zeros(0)
        # self.DM_list = np.zeros(0)
        #self.AM_list = np.zeros(0)
        #self.CM_list = np.zeros(0)
        #self.FM_list = np.zeros(0)

        # plot the second figure
        self.p5, = self.ax02.plot(self.t, self.RI_values, 'r-', linewidth=2.5)
        self.p6, = self.ax02.plot(self.t_prediction, self.RI_predictions, 'o', color='grey', markersize=4, alpha=0.05)
        # self.p6, = self.ax02.plot(self.t, self.HM_list, 'g-', linewidth=2.5)
        # self.p7, = self.ax02.plot(self.t, self.VM_list, 'c-', linewidth=2.5)
        # self.p8, = self.ax02.plot(self.t, self.DM_list, 'm-', linewidth=2.5)
        #self.p9, = self.ax02.plot(self.t, self.AM_list, 'r-', linewidth=2.5)
        #self.p10, = self.ax02.plot(self.t, self.CM_list, 'y-', linewidth=2.5)
        #self.p11, = self.ax02.plot(self.t, self.FM_list, 'k-', linewidth=2.5)

        self.ax02.set_xlabel("Time[sec]")
        self.ax02.set_ylabel("Lifting Risk Index")
        #self.ax02.legend(["RI", "HM", "VM", "DM", "AM", "CM", "FM"])
        # self.ax02.legend(["RI", "HM", "VM", "DM"])
        self.ax02.legend(["LI", "LI predictions"])
        self.ax02.grid(True)

        # self.x = 0.0

        self.timer = current_milli_time()
        # self.counter = 0
        # self.time_length = 100
        self.view_prediction_horizon = 30
        self.time_step = 0.03

        return
    
    def animate(self, dummy):
        print("Timer: {}".format(current_milli_time() - self.timer))
        self.timer = current_milli_time()
        time_now = (current_milli_time() / 1000.0) - self.t0
    #      read_human_jRotPos = human_jPos_port.read()
    # if read_human_jRotPos is not None:
    #     human_jPos_data = []
    #     for i in range(read_human_jRotPos.size()):
    #         human_jPos_data.append(read_human_jRotPos.get(i).asFloat64())
    # print("[INFO] Read human joint positions successfully!")
        # read RI
        read_RI_data = read_RI_port.read(False)
        if read_RI_data is not None:
            RI_tmp = read_RI_data.get(0).asFloat64()
            RI_pred = []
            for i in range(read_RI_data.size()-1):
                RI_pred.append(read_RI_data.get(i+1).asFloat64())
            # single_RI = RI_values.get(0).asFloat64()
            new_time_prediction = [(time_now + i * self.time_step) for i in range(self.view_prediction_horizon)]
            self.t_prediction = append(self.t_prediction, new_time_prediction)
            self.RI_predictions = append(self.RI_predictions, RI_pred)

            # print("[1] RI1: ", self.RI_prediction_df5)
            self.t = append(self.t, time_now)
            self.RI_values = append(self.RI_values, RI_tmp)
        else:
            return self.p5
        
        # print("[INFO] RI length is: ", len(RIs))
        # new_time_prediction = [(time_now + i * self.time_step) for i in range(self.view_prediction_horizon)]
        # self.t_prediction = append(self.t_prediction, new_time_prediction)
        # self.RI_predictions = append(self.RI_predictions, RI_pred)
    
        # print("[1] RI1: ", self.RI_prediction_df5)
        # self.t = append(self.t, time_now)
        # self.RI_values = append(self.RI_values, RI_tmp)

        # self.x += 0.05
        # read NIOSH multipliers
        # NIOSH_factors = read_NIOSH_factors_port.read(False)
        # if NIOSH_factors is not None:
        #     HM_value = NIOSH_factors.get(1).asFloat64()
        #     VM_value = NIOSH_factors.get(2).asFloat64()
        #     DM_value = NIOSH_factors.get(3).asFloat64()
            #AM_value = NIOSH_factors.get(4).asFloat64()
            #CM_value = NIOSH_factors.get(5).asFloat64()
            #FM_value = NIOSH_factors.get(6).asFloat64()
        # else:
            #return self.p6, self.p7, self.p8, self.p9, self.p10, self.p11
            # return self.p6, self.p7, self.p8
        # self.HM_list = append(self.HM_list, HM_value)
        # self.VM_list = append(self.VM_list, VM_value)
        # self.DM_list = append(self.DM_list, DM_value)
        #self.AM_list = append(self.AM_list, AM_value)
        #self.CM_list = append(self.CM_list, CM_value)
        #self.FM_list = append(self.FM_list, FM_value)

        # self.t = append(self.t, time_now)
        # self.x += 0.05
          
        self.p5.set_data(self.t, self.RI_values)
        self.p6.set_data(self.t_prediction, self.RI_predictions)
        # self.p6.set_data(self.t, self.HM_list)
        # self.p7.set_data(self.t, self.VM_list)
        # self.p8.set_data(self.t, self.DM_list)
        #self.p9.set_data(self.t, self.AM_list)
        #self.p10.set_data(self.t, self.CM_list)
        #self.p11.set_data(self.t, self.FM_list)

        if time_now >= self.xmax - self.plot_front_time:
            self.p5.axes.set_xlim(time_now - self.xmax + self.plot_front_time,
                                  time_now + self.plot_front_time)
            self.p6.axes.set_xlim(time_now - self.xmax + self.plot_front_time,
                                  time_now + self.plot_front_time)

            if read_RI_data is not None and (time_now-self.t[0]>self.xmax):
                self.t_prediction = self.t_prediction[self.view_prediction_horizon:]
                self.RI_predictions = self.RI_predictions[self.view_prediction_horizon:]

                self.t = self.t[1:]
                self.RI_values = self.RI_values[1:]
                # self.HM_list = self.HM_list[1:]
                # self.VM_list = self.VM_list[1:]
                # self.DM_list = self.DM_list[1:]
                #self.AM_list = self.AM_list[1:]
                #self.CM_list = self.CM_list[1:]
                #self.FM_list = self.FM_list[1:]
        
        #return self.p5, self.p6, self.p7, self.p8, self.p9, self.p10, self.p11
        return self.p5, self.p6,
plot_niosh = nioshResultsAnimation()

ani_02 = animation.FuncAnimation(plot_niosh.fig2, plot_niosh.animate,
                              interval=20, blit=False, repeat=False)
plt.show()


