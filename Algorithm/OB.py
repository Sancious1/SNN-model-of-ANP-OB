import ENIN

class OB():
    def __init__(self, nENs = 10, nINs = 20, INsPerNeuron=5, action_period = 40):
        self.timestamp = 0  
        self.nENs = nENs
        self.nINs = nINs
        self.INsPerNeuron = nENs*INsPerNeuron        
        self.ENlayer = ENIN.ENlayer(nENs = nENs, nINs = nINs, INsPerNeuron = self.INsPerNeuron)
        self.EN_spike_delay = 1
        self.EN_delivery_time = {}    
        self.gp = action_period  
        self.IN_go = {}  
        self.IN_ids = {}        #stores spiking INs of each action cycle
        self.actionCode = []  
        self.actionSpikes = []                             
        self.spikingINs = []    #from previous timestep
        self.spikingENs = []

    def reset(self):
        self.ENlayer.reset()
        self.spikingINs = []  
        self.spikingENs = []  
 
    def update(self, sensorV, learn_flag, AChLevel=0):
        self.timestamp = self.timestamp + 1 
        #Recording spiking ENs
        for i in self.spikingENs:     
            self.EN_delivery_time[i] = self.EN_spike_delay          
        delayedENspikes = []          
        for i in list(self.EN_delivery_time.keys()):
            if self.EN_delivery_time[i] > 1:
                self.EN_delivery_time[i] = self.EN_delivery_time[i] - 1  
            else:
                del self.EN_delivery_time[i]  
                delayedENspikes.append(i)  
        #Recording spiking INs
        for i in self.spikingINs:    
            self.IN_go[i] = 1  
        if self.timestamp%self.gp == self.gp-2:
            self.IN_ids[self.timestamp+2] = []
            for i in list(self.IN_go.keys()):
                del self.IN_go[i]
                self.IN_ids[self.timestamp+2].append(i)
        #action pulse
        if (self.timestamp-1)%(self.gp/2) == 0: 
            actionPulse = 1 
        else:
            actionPulse = 0  
        #Store action code 
        if (self.timestamp-1)%(self.gp) == 0: 
            self.actionCode.append([0]*self.nENs) 
            self.actionSpikes.append([0]*self.nENs) 
        #Update ENs and INs
        self.spikingENs = self.ENlayer.update(sensorV, self.spikingINs, self.timestamp, learn_flag, actionPulse)
        for i in self.spikingENs:
            self.actionCode[-1][i] = (self.gp/2) - self.timestamp%self.gp + 1 
            self.actionSpikes[-1][i] = self.timestamp  
        return self.spikingENs, self.spikingINs  