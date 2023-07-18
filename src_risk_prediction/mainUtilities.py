# ===============#
#                #
# MAIN UTILITIES #
#                #
# ===============#
import yarp

# configuration for using remapper devices
class confModule(yarp.RFModule, yarp.Property):
    def __init__(self, keyPort, dataPort):
        yarp.RFModule.__init__(self)
        # keyPort is like a "key"
        # could be: humanStatePort, humanDynamicsPort or humanWrenchPort
        self.keyPort = keyPort
        # portPtr stores the string value that pointed by "key" port
        self.portPtr = ""
        self.dataPort = dataPort

    def check(self, options, keyPort):
        self.keyPort = keyPort
        self.portPtr = ""
        if options.check(self.keyPort):
            self.portPtr = options.find(self.keyPort).asString()
            return True
        else:
            return False
    
    # dataPort is defined in e.g. humanStateRemapper.cpp file
    # for checking the configuration options
    def put(self, options, dataPort):
        options.put(dataPort, self.portPtr)
        return True

# log data: joint names (first row)
# from 2nd row: joint positions, joint velocities, joint torques
def logData(fileName, jnames, jpos, jvel, jtorq):
    with open(fileName, 'w') as file:
        # first save joint names in the first row
        for name in jnames:
            if jnames.index(name) == len(jnames)-1:
                file.write(str(name) + '\n')
            else:
                file.write(str(name) + ' ')
        # then save joint positions, velocities and torques 
        for index in range(len(jpos)):
            file.write(str(jpos[index]) + ' ' + str(jvel[index]) + ' ' + 
                       str(jtorq[index]) + '\n')
            

            