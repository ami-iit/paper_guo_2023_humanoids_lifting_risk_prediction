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
read_RWL_port = yarp.BufferedPortBottle()
read_RWL_port.open("/risk_prediction/readRWLValue:i")
RWL_is_connected = yarp.Network.connect("/risk_prediction/recommond_weight_load:o",
                                        "/risk_prediction/readRWLValue:i")
print("[INFO] Read RWL value port is connected: {}".format(RWL_is_connected))

read_NIOSH_port = yarp.BufferedPortBottle()
read_NIOSH_port.open("/risk_prediction/readNIOSHParas:i")
NIOSH_is_connected = yarp.Network.connect("/risk_prediction/niosh_paras:o",                                           
                                          "/risk_prediction/readNIOSHParas:i")
print("[INFO] Read NIOSH paramters port is connected: {}".format(NIOSH_is_connected))

yarp.delay(0.001)

class nioshFactorsAnimation:
    def __init__(self):
        font = {'size': 15}
        mpl.rc('font', **font)
        
        self.xmin = 0.0
        self.xmax = 10.0
        self.plot_front_time = 1.2

        self.fig1 = figure(num=1, figsize=(8, 3.5))
        self.ax01 = self.fig1.subplots()
        self.ax01.set_title("RWL and NIOSH parameters during lifting task", fontsize=16)
        self.ax01.set_ylim(-20, 70)
        self.ax01.set_xlim(self.xmin, self.xmax)

        self.t = np.zeros(0)
        self.t0 = current_milli_time() / 1000.0

        self.RWL_values = np.zeros(0)
        self.H_list = np.zeros(0)
        self.V_list = np.zeros(0)
        self.D_list = np.zeros(0)

        # plot the first figure
        self.p1, = self.ax01.plot(self.t, self.RWL_values, 'b-', linewidth=2.5)
        self.p2, = self.ax01.plot(self.t, self.H_list, 'g-', linewidth=2.5)
        self.p3, = self.ax01.plot(self.t, self.V_list, 'c-', linewidth=2.5)
        self.p4, = self.ax01.plot(self.t, self.D_list, 'm-', linewidth=2.5)
        
        self.ax01.set_xlabel("Time[sec]")
        self.ax01.set_ylabel("RWL[kg] and NIOSH parameters")
        self.ax01.legend(["RWL", "H", "V", "D"])
        self.ax01.grid(True)

        # self.x = 0.0

        self.timer = current_milli_time()
        self.counter = 0
        # self.time_length = 100

        return
    
    def animate(self, dummy):
        print("Timer: {}".format(current_milli_time() - self.timer))
        self.timer = current_milli_time()
        time_now = (current_milli_time() / 1000.0) - self.t0

        # read RWL
        RWL_values = read_RWL_port.read(False)
        if RWL_values is not None:
            single_RWL = RWL_values.get(0).asFloat64()
        else:
            return self.p1
        self.RWL_values = append(self.RWL_values, single_RWL)
        
        # read NIOSH parameters
        NIOSH_values = read_NIOSH_port.read(False)
        if NIOSH_values is not None:
            H_value = NIOSH_values.get(1).asFloat64()
            V_value = NIOSH_values.get(2).asFloat64()
            D_value = NIOSH_values.get(3).asFloat64()
        else:
            return self.p2, self.p3, self.p4
        self.H_list = append(self.H_list, H_value)
        self.V_list = append(self.V_list, V_value)
        self.D_list = append(self.D_list, D_value)

        self.t = append(self.t, time_now)
        # self.x += 0.05
          
        self.p1.set_data(self.t, self.RWL_values)
        self.p2.set_data(self.t, self.H_list)
        self.p3.set_data(self.t, self.V_list)
        self.p4.set_data(self.t, self.D_list)

        if time_now >= self.xmax - self.plot_front_time:
            self.p1.axes.set_xlim(time_now - self.xmax + self.plot_front_time,
                                  time_now + self.plot_front_time)
            self.p2.axes.set_xlim(time_now - self.xmax + self.plot_front_time,
                                  time_now + self.plot_front_time)
            self.p3.axes.set_xlim(time_now - self.xmax + self.plot_front_time,
                                  time_now + self.plot_front_time)
            self.p4.axes.set_xlim(time_now - self.xmax + self.plot_front_time,
                                  time_now + self.plot_front_time)
            if RWL_values is not None:
                self.t = self.t[1:]
                self.RWL_values = self.RWL_values[1:]
                self.H_list = self.H_list[1:]
                self.V_list = self.V_list[1:]
                self.D_list = self.D_list[1:]
               
        
        return self.p1, self.p2, self.p3, self.p4

plot_niosh = nioshFactorsAnimation()

ani_01 = animation.FuncAnimation(plot_niosh.fig1, plot_niosh.animate,
                              interval=20, blit=False, repeat=False)
plt.show()

