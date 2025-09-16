import numpy as np

class Integrator:
    def __init__(self):
        self.V = 0
        self.spikeTheta = 20
        self.w = {}
        self.spiketimes = []
        self.monitorFlag = 0
        self.vMonitor = [0]
        self.rperiod = 20
        self.rcounter = 0

    def update(self, timestamp, Vin = 0):
        if self.rcounter == 0:
            self.V = self.V + Vin
            if self.monitorFlag == 1:
                self.vMonitor.append(self.V)
            if self.V >= self.spikeTheta:
                self.spiketimes.append(timestamp)
                self.V = 0
                self.rcounter = self.rperiod
        else:
            self.rcounter = self.rcounter - 1

class ENIN():
    def __init__(self, ID):
        self.ID = ID    
        self.AD = Integrator()
        self.actionState = 0    
        self.inhibitoryWeights = {}   
        self.plasticSynapses = []    
        self.spiketimes = []            
        self.rperiod = 20    
        self.rcounter = 0    
        self.inhLearningRate = 1.0         
        self.blockingInhibitions = {}      #keys are IDs of IN->EN synapses that are in their inhibitory states values are counters
        self.reboundInhibitions = []    
        self.ADspikeTime = 0    
        self.inhReleaseTimes = {}    
        self.ADtrigger = 0    
        self.PSPsum = 0    
        self.monitorFlag = 0    
        self.wMonitor = []    
        self.synMonitor = [0]   
        self.inhMonitor = [0]   
        self.stateMonitor = [0]   
        self.stateMonitor2 = [0]   
        
    def reset(self): 
        self.blockingInhibitions.clear()    #clear all elements in blockingInhibitions
        self.reboundInhibitions = []    
        self.inhReleaseTimes.clear()     #clear all elements in inhReleaseTimes
        self.PSPsum = 0            
        
    def monitor(self):
        self.wMonitor.append(self.inhibitoryWeights.values())        #Assign key-value in dictionary inhibitoryWeights to wMonitor
        
    def initSynapse(self, w_index):
        self.inhibitoryWeights[w_index] = 0   
        self.plasticSynapses.append(w_index)     #Assign w_index to plasticSynapses
    
    def updateactionState(self, learn_flag=0): 
        if self.actionState == 1: 
            self.actionState = 0               
            self.AD.V = 0     #Neuron membrane AD = neuron.integrator()
            self.ADtrigger  = 0    
            if(learn_flag==1): 
                self.updateInhWeights()       #If learn_flag=1, weights update and execute function updateInhWeights
            self.inhReleaseTimes.clear()      #clear all elements in inhReleaseTimes
            if(learn_flag==1 and self.monitorFlag==1):
                self.monitor()
        else:   
            self.actionState= 1              #If learn_flag=0, there is no weights updating and actionState=1
            self.ADspikeTime = 0              # The time of neuron spiking

    def updateInhibitoryStates(self, localINspikes, timestamp):    
        self.reboundInhibitions = []                 
        for i in list(self.blockingInhibitions.keys()):   # blockingInhibitions' keys are IDs of IN-EN synapses
            self.blockingInhibitions[i] = self.blockingInhibitions[i] - 1   
            if(self.blockingInhibitions[i]<=0):         
                del self.blockingInhibitions[i]   
                self.PSPsum = self.PSPsum + 1    
                self.inhReleaseTimes[i] = timestamp   
                if(self.inhibitoryWeights[i] > 2):      #if sufficient inhibition>2, it is saturated
                    self.reboundInhibitions.append(i)     
        for i in self.inhibitoryWeights.keys():
            if i in localINspikes:
                self.blockingInhibitions[i] = self.inhibitoryWeights[i]   
                self.PSPsum = self.PSPsum - 1    
        
    def generateSpike(self, timestamp): 
        self.spiketimes.append(timestamp)     
        self.rcounter = self.rperiod    
        return 1    

    def update(self, sensorV, spiking_INs, timestamp, learn_flag, actionPulse):
        #spiking_INs -- list of indices of INs that spiked in the previous timestep
        if actionPulse == 1:                                
            self.updateactionState(learn_flag)                
        #Update IN inhibition states
        self.updateInhibitoryStates(spiking_INs, timestamp)           
        #Update apical dendrite
        ADin = 0    
        if self.actionState==1: 
            ADin = sensorV   
        self.AD.update(timestamp, Vin=ADin)
        if (timestamp in self.AD.spiketimes):
            self.ADspikeTime = timestamp   
            self.ADtrigger = 1   
        #Update Soma
        spikeFlag = 0   
        if(learn_flag==0):
            sumV = self.ADtrigger + self.PSPsum + len(self.reboundInhibitions)
        else:
            sumV = self.ADtrigger                 
        if(self.rcounter<=0):
            if(sumV>0 and self.actionState==1):
                spikeFlag = self.generateSpike(timestamp)   
        else:
            self.rcounter = self.rcounter - 1            
        return spikeFlag    

    def updateInhWeights(self):
        k = self.inhLearningRate    
        for i in self.inhReleaseTimes.keys(): 
            if i in self.plasticSynapses:
                if(self.ADspikeTime!=0):     #if AD spike 
                    delw = k*(self.ADspikeTime - self.inhReleaseTimes[i])   
                else:
                    delw = 40         
                self.inhibitoryWeights[i] = np.clip((self.inhibitoryWeights[i] + delw), 0, 40)    
            
class ENlayer():
    def __init__(self, nENs, nINs, INsPerNeuron):
        self.ENs = []   
        INsPerNeuronPerEN = INsPerNeuron//nENs   
        nNeuron = nINs//INsPerNeuron    
        for i in range(0, nENs):
            self.ENs.append(ENIN(ID=i))
            for j in range(0, nNeuron):
                INidStart = j*INsPerNeuron+i*INsPerNeuronPerEN    
                INidStop = INidStart + INsPerNeuronPerEN   
                for k in range(INidStart, INidStop):
                    self.ENs[i].initSynapse(k)   

    def reset(self):
        for i in range(0, len(self.ENs)):
            self.ENs[i].reset()    

    def update(self, sensorV, spiking_INs, timestamp, learn_flag, actionPulse):
        spikingENs = []    
        for i in range(0,len(self.ENs)):
            spikeFlag = self.ENs[i].update(sensorV[i], spiking_INs, timestamp, learn_flag, actionPulse)
            if spikeFlag == 1: 
                spikingENs.append(i)   
        return spikingENs    