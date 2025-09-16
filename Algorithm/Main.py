import OB
import numpy as np
import pickle

#Load Data
rf = open('./UCSD_Batch1.pi', 'rb')
trainingset = np.array(pickle.load(rf))
testset = np.array(pickle.load(rf))
train_labels = np.array(pickle.load(rf))
test_labels = np.array(pickle.load(rf))
rf.close()

nset = len(trainingset)
nTestPerGas = len(testset)/nset
print("Number of set to train = " + str(len(trainingset)))
print("Number of set to test = " + str(len(testset)))

#Network initialization
nENs = len(trainingset[0])
INsPerNeuron = 5
nINs = nENs*INsPerNeuron*nset
OB = OB.OB(nENs, nINs, INsPerNeuron)

#Encoding
def encoder(Gas=[]):
    Enco = []
    encoding = {0: 0, 1: 1.0, 2: 1.06, 3: 1.12, 4: 1.18, 5:1.25, 6:1.34, 7:1.43, 8:1.54, 9:1.67, 10:1.82, 11:2.0, 12:2.3, 13:2.5, 14:2.86, 15:3.34}
    for i in range(0, len(Gas)):
        Enco.append(encoding[Gas[i]])
    return Enco

#genPattern
def genPattern(Gas, learn_flag=0, nactionPerGas=5, actPeriod=40):
    sensorInput = encoder(Gas)
    for j in range(0, nactionPerGas):
        for k in range(0, actPeriod):
            OB.update(sensorInput, learn_flag=learn_flag)
            pass
    OB.reset()
#Classifier
def similarity(l1, l2):
    list1 = []
    list2 = []
    for i in range(0, len(l1)):
        list1.append((i, l1[i]))
        list2.append((i, l2[i]))
    set1 = set(list1)
    set2 = set(list2)
    intersectionSize = len(set.intersection(set1, set2))
    unionSize = len(set.union(set1, set2))
    return intersectionSize / float(unionSize)

def TrainedCode(actionCode, nGas, nactionPerLearning):
    learnedactionCode = []
    for i in range(1, nGas + 1):
        labelactionID = i * nactionPerLearning + 5 * i - 1
        learnedactionCode.append(actionCode[labelactionID])
    return learnedactionCode

def findSImatrixaction(actionCode, learnedactionCode, nGas, testStartID):
    SImatrixaction = []
    k = 0
    for i in range(testStartID, len(actionCode)):
        SImatrixaction.append([])
        for j in range(0, nGas):
            smlt = similarity(actionCode[i], learnedactionCode[j])
            SImatrixaction[k].append(round(smlt, 2))
        k += 1
    return SImatrixaction

def classifier(actionCode, nGas):
    nTrainSamplesPerGas = 1
    nactionPerLearning = nTrainSamplesPerGas * 5
    learnedactionCode = TrainedCode(actionCode, nGas, nactionPerLearning)
    testStartID = nGas * nactionPerLearning + nGas * 5
    SImatrix = findSImatrixaction(actionCode, learnedactionCode, nGas, testStartID)
    return SImatrix

#Training
for i in range(0, len(trainingset)):
    print("Training Gas " + str(i+1))
    genPattern(trainingset[i], learn_flag=1)
    genPattern(trainingset[i], learn_flag=0)

#Testing
for i in range(0, len(testset)):
    genPattern(testset[i], learn_flag=0)
    if(i%10==0 and i!=0):
        print(str(i) + " set tested")

#Classifying
sMatrix = classifier(OB.actionCode, nset)

sMatrix_eff = []
for i in range(0,len(sMatrix),5):
    sMatrix_eff.append(sMatrix[i+4])

right = 0 
wrong = 0 

for i in range(0,len(sMatrix_eff)):
    print('Test sample',i+1,'Recognition results:',train_labels[sMatrix_eff[i].index(max(sMatrix_eff[i]))],'Actual results',test_labels[i])
    if (train_labels[sMatrix_eff[i].index(max(sMatrix_eff[i]))]) == test_labels[i]:
        right = right + 1
    else:
        wrong = wrong + 1

print('Test accuracy:',(right/len(test_labels))*100,'%')


