import numpy as np
import copy
import pickle

def maxtomin(data):
    nSensors = len(data[0]) 
    range = [[0,0]]*nSensors
    for i in range(0, len(data)):
        for j in range(0, nSensors):  
            if(i==0):
                range[j] = [data[i][j], data[i][j]]
            elif data[i][j] < range[j][0]:
                range[j][0] = data[i][j]
            elif data[i][j] > range[j][1]:
                range[j][1] = data[i][j]
    return range

def gap(data, nBins):
    range = maxtomin(data)
    gapValue = []
    for i in range:
        interval = i[1]-i[0]
        gapValue.append(round((interval)/float(nBins-1), 2))
    return range, gapValue

def postdata(gasOrigin, gapValue, range, nBins):
    gasMain = []
    for i in range(0, len(gasOrigin)):
        gasMain.append([])
        for j in range(0, len(range)):
            temp = (gasOrigin[i][j] - range[j][0])/gapValue[j]
            temp = np.clip(int(round(temp)), 0, nBins-1)
            gasMain[i].append(temp)  
    return gasMain

def sparse(gasDense, num_sensors):
    top = [0]* num_sensors            #list of most active sensors
    gasTemp = copy.deepcopy(gasDense)  
    cutoff = 64         #number of sensors that make the top list
    for i in range(0, cutoff):
        m = max(gasTemp) 
        index1 = gasTemp.index(m)  
        gasTemp[index1] = 0 
        top[index1] = m  
    return top  

def sparseData(gassDense, num_sensors):
    gasSparsified = []
    for i in gassDense:
        s = sparse(i, num_sensors)
        gasSparsified.append(s)
    return gasSparsified

def package(file_name, train_set, test_set, train_labels, test_labels):
    with open(file_name, 'wb') as wf:
        pickle.dump(train_set, wf)
        pickle.dump(test_set, wf)
        pickle.dump(train_labels, wf)
        pickle.dump(test_labels, wf)