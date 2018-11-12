import numpy as np
import matplotlib.pyplot as plt
#from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import os
import warnings
import scipy.stats as sp
import cv2
import re,traceback,xlwt
warnings.filterwarnings("ignore")
#####################################################################################################
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Testing Results")
sheet1.write(0, 0, "S.NO")
sheet1.write(0, 1, "File Name")
sheet1.write(0, 2, "Organization Name")
sNo=1
# Defining Hyper Parameters
cell_size = (8, 8)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

#########################################################################################################
def HOG(img,cell_size,block_size,nbins):
    try:
        # Initialzing HOG Descriptor
        # winSize is the size of the image cropped to an multiple of the cell size
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)

        n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
        hog_feats = hog.compute(img)\
                       .reshape(n_cells[1] - block_size[1] + 1,
                                n_cells[0] - block_size[0] + 1,
                                block_size[0], block_size[1], nbins) \
                       .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
        # hog_feats now contains the gradient amplitudes for each direction,
        # for each cell of its group for each group. Indexing is by rows then columns.

        gradients = np.zeros((n_cells[0], n_cells[1], nbins))

        # count cells (border cells appear less often across overlapping groups)
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

        for off_y in range(block_size[0]):
            for off_x in range(block_size[1]):
                gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                          off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                           off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

        # Average gradients
        gradients /= cell_count
        return(gradients)
    except:
        print("Error in HOG", traceback.print_exc())
#########################################################################################################
fpaths = []
labels = []
spoken = []
features = []
ClassSamples=[]
for f in os.listdir('Dataset'):
    ClassSamples.append(len(os.listdir('Dataset/' + f)))
    if f not in spoken:
        spoken.append(f)

print('Class:', spoken)
print("Samples Per Class:",ClassSamples)
print("..............................................................")
trainingSamples=[None]*len(spoken)
testingSamples=[None]*len(spoken)
population=None
#####################################################################################################
for f in os.listdir('Dataset'):
    print("Reading...",f)
    for w in os.listdir('Dataset/' + f):
        Sum=0
        labels.append(f)
        for idx,v in enumerate(os.listdir('Dataset/' + f + '/' + w)):
            #print('Dataset/' + f + '/' + w+'/' + v)
            fileName='Dataset/' + f + '/' + w+'/' + v
            fpaths.append(fileName)
            ##############################################
            img = cv2.cvtColor(cv2.imread(fileName),cv2.COLOR_BGR2GRAY)
            # Image Normalization
            
            featuresHOG=HOG(img,cell_size,block_size,nbins)
            Sum=Sum+featuresHOG
            ##############################################
        Sum=Sum/len(os.listdir('Dataset/' + f))
        print(Sum.shape)
        A=list(Sum.reshape((Sum.shape[0]*Sum.shape[1]*Sum.shape[2])))
        features.append(A)

print("Labels...",len(labels))
print("Samples...",len(features))
print("..............................................................")
#####################################################################################################
#print(features)
c = list(zip(features, labels))
#print(c)
np.random.shuffle(c)
features,labels = zip(*c)
print("Population...")
print("Samples...",len(features))
print("Labels...",len(labels))
#print(features[0])

m_trainingsetfeatures = features[0:20]
m_trainingsetlabels = labels[0:20]

print("Training Samples...",len(m_trainingsetfeatures))
print("Training Labels...",len(m_trainingsetlabels))
#print(m_trainingsetlabels)
m_testingsetfeatures = features[20:28]
m_testingsetlabels = labels[20:28]


print("Testing Samples...",len(m_testingsetfeatures))
print("Testing Labels...",len(m_testingsetlabels))
print("..............................................................")
#####################################################################################################

gmmhmmindexdict = {}
index = 0
for word in spoken:
    gmmhmmindexdict[word] = index
    index = index +1


print ('Loading of data completed')

#Parameters needed to train GMMHMM
m_num_of_HMMStates = 3  # number of states
m_num_of_mixtures = 2  # number of mixtures for each hidden state
m_covarianceType = 'diag'  # covariance type
m_n_iter = 10  # number of iterations
m_bakisLevel = 2


def initByBakis(inumstates, ibakisLevel):
    startprobPrior = np.zeros(inumstates)
    startprobPrior[0: ibakisLevel - 1] = 1/float((ibakisLevel - 1))
    transmatPrior = getTransmatPrior(inumstates, ibakisLevel)
    return startprobPrior, transmatPrior


def getTransmatPrior(inumstates, ibakisLevel):
    transmatPrior = (1 / float(ibakisLevel)) * np.eye(inumstates)

    for i in range(inumstates - (ibakisLevel - 1)):
        for j in range(ibakisLevel - 1):
            transmatPrior[i, i + j + 1] = 1. / ibakisLevel

    for i in range(inumstates - ibakisLevel + 1, inumstates):
        for j in range(inumstates - i - j):
            transmatPrior[i, i + j] = 1. / (inumstates - i)

    return transmatPrior

m_startprobPrior ,m_transmatPrior = initByBakis(m_num_of_HMMStates,m_bakisLevel)


print("StartProbPrior=")
print(m_startprobPrior)

print("TransMatPrior=")
print(m_transmatPrior)


class SpeechModel:
    def __init__(self,Class,label):
        self.traindata = np.zeros((0,6))
        self.Class = Class
        self.label = label
        #self.model  = hmm.GMMHMM(n_components = m_num_of_HMMStates, n_mix = m_num_of_mixtures, \
        #                   transmat_prior = m_transmatPrior, startprob_prior = m_startprobPrior, \
        #                                covariance_type = m_covarianceType, n_iter = m_n_iter)
        self.model  = hmm.GMMHMM(n_components = m_num_of_HMMStates, covariance_type = m_covarianceType, n_iter = m_n_iter)



#7 GMMHMM Models would be created for 7 words
speechmodels = [None] * len(spoken)

for key in gmmhmmindexdict:
    speechmodels[gmmhmmindexdict[key]] = SpeechModel(gmmhmmindexdict[key],key)

print("Train Data...",speechmodels[0].traindata,"Class...",speechmodels[0].Class,"Label...",speechmodels[0].label)
print("Training Samples...",len(m_trainingsetfeatures))
print("Gesture Models...",len(speechmodels))
print("..............................................................")
print("Training Started...")
for i in range(0,len(m_trainingsetfeatures)):
     for j in range(0,len(speechmodels)):
         if int(speechmodels[j].Class) == int(gmmhmmindexdict[m_trainingsetlabels[i]]):
            T=np.array(m_trainingsetfeatures[i])
            if len(speechmodels[j].traindata)==0:
                speechmodels[j].traindata=T
            else:
                speechmodels[j].traindata=np.row_stack((speechmodels[j].traindata,T))
            #speechmodels[j].traindata = np.concatenate((m_trainingsetfeatures[i]))
            #speechmodels[j].traindata = np.column_stack(m_trainingsetfeatures[i])
          


for speechmodel in speechmodels:
    #print(speechmodel.traindata)
    #print(len(speechmodel.traindata),(speechmodel.traindata).shape)
    speechmodel.model.fit(speechmodel.traindata)


print ('Training completed -- '+str(len(spoken))+' GMM-HMM models are built for '+str(len(spoken))+' different types of Gestures')
print("..............................................................")
#####################################################################################################
print("Prediction started")

#Testing
m_PredictionlabelList = []

for i in range(0,len(m_testingsetfeatures)):
    scores = []
    for speechmodel in speechmodels:
         T=np.array(m_testingsetfeatures[i])
         T=np.reshape(T, (1, len(m_testingsetfeatures[i]))) 
         #print(T.shape)
         scores.append(speechmodel.model.score(T))
    id  = scores.index(max(scores))
    m_PredictionlabelList.append(speechmodels[id].Class)
    print("Test-",i,str(np.round(scores, 3)) + " " + str(max(np.round(scores, 3))) +" "+":"+ speechmodels[id].label)
    sheet1.write(maxscore, 0, str(max(np.round(scores, 3))))

accuracy = 0.0
count = 0
print("..............................................................")

print("")
print("Prediction for Testing DataSet:")

for i in range(0,len(m_testingsetlabels)):
    if gmmhmmindexdict[m_testingsetlabels[i]] == m_PredictionlabelList[i]:
       count = count+1
       print( "Actual Label:"+m_testingsetlabels[i],"\tPrediction:"+spoken[m_PredictionlabelList[i]],"\t-> Match")
    else:
       print( "Actual Label:"+m_testingsetlabels[i],"\tPrediction:"+spoken[m_PredictionlabelList[i]],"\t-> No Match")


accuracy = 100.0*count/float(len(m_testingsetlabels))
print("..............................................................")
print("")
print("Total Test Samples:",len(m_testingsetlabels))
print("Correctly Classified:",count)
print("accuracy ="+str(accuracy))
print("")









