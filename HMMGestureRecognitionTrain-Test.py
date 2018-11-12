import librosa
import numpy as np
import matplotlib.pyplot as plt
#from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import os
import cv2
import warnings
import scipy.stats as sp
import xlwt
import traceback
warnings.filterwarnings("ignore")

# Dumping Results into the XL
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Training Results")
sheet1.write(0, 0, "Sample#")
sheet1.write(0, 1, "Likelihood")
sheet1.write(0, 2, "Predicted Label")
sheet1.write(0, 3, "Actual Label")
sheet2 = book.add_sheet("Testing Results")
sheet2.write(0, 0, "Sample#")
sheet2.write(0, 1, "Likelihood")
sheet2.write(0, 2, "Predicted Label")
sheet2.write(0, 3, "Actual Label")

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
########################################################################################################## Training Population
fpaths = [];trainLabels = [];features=[];trainClasses = [];trainClassPopulation=[]
for f in os.listdir('Training'):
    print("Reading...",f,end="")
    trainClassPopulation.append(len(os.listdir('Training/' + f)))
    for w in os.listdir('Training/' + f):
        Sum=0
        trainLabels.append(f)
        if f not in trainClasses:
            trainClasses.append(f)
        for idx,v in enumerate(os.listdir('Training/' + f + '/' + w)):
            #print('Dataset/' + f + '/' + w+'/' + v)
            fileName='Training/' + f + '/' + w+'/' + v
            fpaths.append(fileName)
            ##############################################
            img = cv2.cvtColor(cv2.imread(fileName),cv2.COLOR_BGR2GRAY)
            featuresHOG=HOG(img,cell_size,block_size,nbins)
            #Temp=list(featuresHOG.reshape((featuresHOG.shape[0]*featuresHOG.shape[1]*featuresHOG.shape[2])))
            if idx==0:
                Sum=featuresHOG
            else:
                Sum=np.add(Sum, featuresHOG)
            ##############################################
        Sum=np.divide(Sum,len(os.listdir('Training/' + f)))
        A=list(Sum.reshape((Sum.shape[0]*Sum.shape[1]*Sum.shape[2])))
        features.append(A)
    print("\t\t... DONE!")
print("..............................................................")
#####################################################################################################print('Words spoken:', trainClasses)
print("Samples Per Class:",trainClassPopulation)
print("..............................................................")
c = list(zip(features, trainLabels))
m_trainingsetfeatures,m_trainingsetlabels = zip(*c)
print("Training Samples...",len(m_trainingsetfeatures))
print("Training Labels...",len(m_trainingsetlabels))
#####################################################################################################
# Testing Population
fpaths = [];testLabels = [];testClasses = [];features=[];testClassPopulation=[]
for f in os.listdir('Testing'):
    print("Reading...",f,end="")
    testClassPopulation.append(len(os.listdir('Testing/' + f)))
    for w in os.listdir('Testing/' + f):
        Sum=0
        testLabels.append(f)
        if f not in testClasses:
            testClasses.append(f)
        for idx,v in enumerate(os.listdir('Testing/' + f + '/' + w)):
            #print('Dataset/' + f + '/' + w+'/' + v)
            fileName='Testing/' + f + '/' + w+'/' + v
            fpaths.append(fileName)
            ##############################################
            img = cv2.cvtColor(cv2.imread(fileName),cv2.COLOR_BGR2GRAY)
            featuresHOG=HOG(img,cell_size,block_size,nbins)
            #Temp=list(featuresHOG.reshape((featuresHOG.shape[0]*featuresHOG.shape[1]*featuresHOG.shape[2])))
            if idx==0:
                Sum=featuresHOG
            else:
                Sum=np.add(Sum, featuresHOG)
            ##############################################
        Sum=np.divide(Sum,len(os.listdir('Testing/' + f)))
        A=list(Sum.reshape((Sum.shape[0]*Sum.shape[1]*Sum.shape[2])))
        features.append(A)
    print("\t\t... DONE!")
print("..............................................................")
print('Words spoken:', testClasses)
print("Samples Per Class:",testClassPopulation)
print("..............................................................")
# Reading Training features
c = list(zip(features, testLabels))
m_testingsetfeatures,m_testingsetlabels = zip(*c)
print("Testing Samples...",len(m_testingsetfeatures))
print("Testing Labels...",len(m_testingsetlabels))

#####################################################################################################
gmmhmmindexdict = {}
index = 0
for word in trainClasses:
    gmmhmmindexdict[word] = index
    index = index +1
print ('Loading of data completed')

#Parameters needed to train GMMHMM
m_num_of_HMMStates = 3  # number of states
m_num_of_mixtures = 2  # number of mixtures for each hidden state
m_covarianceType = 'diag'  # covariance type
m_n_iter = 20  # number of iterations
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
speechmodels = [None] * len(trainClasses)

for key in gmmhmmindexdict:
    speechmodels[gmmhmmindexdict[key]] = SpeechModel(gmmhmmindexdict[key],key)

print("Train Data...",speechmodels[0].traindata,"Class...",speechmodels[0].Class,"Label...",speechmodels[0].label)
print("Training Samples...",len(m_trainingsetfeatures))
print("Speech Models...",len(speechmodels))

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


print ('Training completed --  GMM-HMM models are built for  different types of words')
print("..............................................................")
#####################################################################################################
# Training accuracy
minValue=-1720;maxValue=80
print("..............................................................")
print("")
print("Prediction for Training DataSet:")
likelihoods=[];accuracy = 0.0;count = 0;sNo=1
for i in range(0,len(m_trainingsetfeatures)):
    scores = []
    for speechmodel in speechmodels:
         T=np.array(m_trainingsetfeatures[i])
         T=np.reshape(T, (1, len(m_trainingsetfeatures[i])))
         #print(T.shape)
         scores.append(speechmodel.model.score(T))
    id  = scores.index(max(scores))
    #print("Test-",i,str(np.round(scores, 3)) + " " + str(max(np.round(scores, 3))) +" "+":"+ speechmodels[id].label)
    likelihoods.append(max(np.round(scores, 3)))
    normalized = (max(np.round(scores, 3))-minValue)/(maxValue-minValue)
    # Prediction Summary
    if gmmhmmindexdict[m_trainingsetlabels[i]] == speechmodels[id].Class:
        count = count+1
        print( "Actual Label:"+m_trainingsetlabels[i],"\tPrediction:"+trainClasses[speechmodels[id].Class],"\t-> Match", "\t Likelihood:",str(normalized))
    else:
        print( "Actual Label:"+m_trainingsetlabels[i],"\tPrediction:"+trainClasses[speechmodels[id].Class],"\t-> No Match", "\t Likelihood:",str(normalized))
    # Dump Dataset
    sheet1.write(sNo, 0, sNo);sheet1.write(sNo, 1, normalized);sheet1.write(sNo, 2, trainClasses[speechmodels[id].Class]);sheet1.write(sNo, 3, m_trainingsetlabels[i])
    sNo=sNo+1
#print(scores)
accuracy = 100.0*count/float(len(m_trainingsetlabels))
print("..............................................................")
print("")
print("Total Training Samples:",len(m_trainingsetlabels))
print("Correctly Classified:",count)
print("accuracy ="+str(accuracy))
print("")

# Testing Started
print("Prediction started.... Testing Samples")
print("..............................................................")
print("")
print("Prediction for Testing DataSet:")
accuracy = 0.0;count = 0;sNo=1
for i in range(0,len(m_testingsetfeatures)):
    scores = []
    for speechmodel in speechmodels:
         T=np.array(m_testingsetfeatures[i])
         T=np.reshape(T, (1, len(m_testingsetfeatures[i])))
         #print(T.shape)
         scores.append(speechmodel.model.score(T))
    id  = scores.index(max(scores))
    #print("Test-",i,str(np.round(scores, 3)) + " " + str(max(np.round(scores, 3))) +" "+":"+ speechmodels[id].label)
    likelihoods.append(max(np.round(scores, 3)))
    normalized = (max(np.round(scores, 3))-minValue)/(maxValue-minValue)
    if gmmhmmindexdict[m_testingsetlabels[i]] == speechmodels[id].Class:
        count = count+1
        print( "Actual Label:"+m_testingsetlabels[i],"\tPrediction:"+testClasses[speechmodels[id].Class],"\t-> Match", "\t Likelihood:",str(normalized))
    else:
        print( "Actual Label:"+m_testingsetlabels[i],"\tPrediction:"+testClasses[speechmodels[id].Class],"\t-> No Match", "\t Likelihood:",str(normalized))
    # Dump Dataset
    sheet2.write(sNo, 0, sNo);sheet2.write(sNo, 1, normalized);sheet2.write(sNo, 2,testClasses[speechmodels[id].Class]);sheet2.write(sNo, 3,  m_testingsetlabels[i])
    sNo=sNo+1
accuracy = 100.0*count/float(len(m_testingsetlabels))
print("..............................................................")
print("")
print("Total Test Samples:",len(m_testingsetlabels))
print("Correctly Classified:",count)
print("accuracy ="+str(accuracy))
print("")
print("..............................................................")
print(likelihoods,min(likelihoods),max(likelihoods))
book.save("TestingResults.xls")
