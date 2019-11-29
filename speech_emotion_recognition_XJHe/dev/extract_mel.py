#!/usr/bin/env python
# coding: utf-8

# In[33]:


"""
@author: daehyeon
"""

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import cPickle
import random
import csv


import warnings
warnings.simplefilter("ignore", DeprecationWarning)


# In[34]:


#parameter
subsegLen = 300
filter_num = 40

#wav files dir for training and validation
trainWavFolder = "/mnt/home/20150132/Desktop/dataset/wav/train/"
valWavFolder = "/mnt/home/20150132/Desktop/dataset/wav/val/"
testWavFolder = "/mnt/home/20150132/Desktop/dataset/wav/test1/"

def getFilesList(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

trainWavfiles = getFilesList(trainWavFolder)
valWavfiles = getFilesList(valWavFolder)
testWavfiles = getFilesList(testWavFolder)


# In[35]:


toLabel = {'hap':0, 'sur':1, 'dis':2, 'neu':3, 'fea':4, 'sad':5, 'ang':6}


# In[36]:



def read_wav_file(filename):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    
    return wavedata, time, framerate

def load_zscore():
    f = open('./zscore40.pkl','rb')
    mean1,std1,mean2,std2,mean3,std3 = cPickle.load(f)
    return mean1,std1,mean2,std2,mean3,std3


# In[37]:


def dense_to_one_hot(labels_dense, num_classes = 7):
    """
    Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot


# In[30]:


def next_batch(num, train = True):
    eps = 1e-5 

    dataList = []
    labelList = []
    subsegNumPerFile = np.empty(num, dtype = np.int8) # we need to know how many subsegments are in each of file to calculate accuracy.

    # load zscore data 
    melMean,melStd,delta1Mean,delta1Std,delta2Mean,delta2Std = load_zscore()

    if(train):
        rdFiles = random.sample(range(len(trainWavfiles)), num) #randomly choose 'num' of files from wavfiles
    else:
        rdFiles = random.sample(range(len(valWavfiles)), num) #randomly choose 'num' of files from wavfiles

    index = 0
    for rdFile in rdFiles:
        if(train):
            fileName = trainWavFolder+trainWavfiles[rdFile]
            emotion = trainWavfiles[rdFile][21:24]
        else:
            fileName = valWavFolder+valWavfiles[rdFile]
            emotion = valWavfiles[rdFile][21:24]

        data, _, rate = read_wav_file(fileName)

        label = toLabel[emotion]

        # get the log-mel value with delta and delta delta
        mel_spec = ps.logfbank(data,rate*2,nfilt = filter_num, nfft=1024)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)

        time = mel_spec.shape[0] # the time of audio

        frames = time/subsegLen + 1

        subsegNumPerFile[index] = frames

        for i in range(frames):
            begin = subsegLen*i
            if(frames-1 == i): # for the last frame
                end =time
            else: 
                end= begin + subsegLen

            # subsegmentize
            melt = mel_spec[begin:end]
            delta1t = delta1[begin:end]
            delta2t = delta2[begin:end]

            # if last subsegment, then pad zeros and make it 3 seconds.
            if(begin+ subsegLen != end):
                melt = np.pad(melt, ((0, subsegLen - melt.shape[0]),(0,0)), 'constant', constant_values = 0)
                delta1t = np.pad(delta1t, ((0, subsegLen - delta1t.shape[0]),(0,0)), 'constant', constant_values = 0)
                delta2t = np.pad(delta2t, ((0, subsegLen - delta2t.shape[0]),(0,0)), 'constant', constant_values = 0)

            eachData = np.empty((300, filter_num , 3), dtype = np.float32)
            # Normalize
            eachData[:,:,0] = (melt - melMean)/(melStd+eps)
            eachData[:,:,1] = (delta1t - delta1Mean)/(delta1Std+eps)
            eachData[:,:,2] = (delta2t - delta2Mean)/(delta2Std+eps)
            eachData = np.expand_dims(eachData, axis = 0)

            dataList.append(eachData)
            labelList.append(label)

        index+=1

    wholeData = np.vstack(dataList)
    wholeLabel = dense_to_one_hot(np.array(labelList))

    return wholeData, wholeLabel, subsegNumPerFile


# In[31]:


def test_dataset(testWavFolder,testWavfiles, index, num = 45):
    eps = 1e-5 

    dataList = []
    subsegNumPerFile = np.empty(num, dtype = np.int8) # we need to know how many subsegments are in each of file to calculate accuracy.

    # load zscore data 
    melMean,melStd,delta1Mean,delta1Std,delta2Mean,delta2Std = load_zscore()
    
    for j in range(num):
        
        fileName = testWavFolder+testWavfiles[index+j]

        data, _, rate = read_wav_file(fileName)

        # get the log-mel value with delta and delta delta
        mel_spec = ps.logfbank(data,rate*2,nfilt = filter_num, nfft=1024)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)

        time = mel_spec.shape[0] # the time of audio

        frames = time/subsegLen + 1

        subsegNumPerFile[j] = frames

        for i in range(frames):
            begin = subsegLen*i
            if(frames-1 == i): # for the last frame
                end =time
            else: 
                end= begin + subsegLen

            # subsegmentize
            melt = mel_spec[begin:end]
            delta1t = delta1[begin:end]
            delta2t = delta2[begin:end]

            # if last subsegment, then pad zeros and make it 3 seconds.
            if(begin+ subsegLen != end):
                melt = np.pad(melt, ((0, subsegLen - melt.shape[0]),(0,0)), 'constant', constant_values = 0)
                delta1t = np.pad(delta1t, ((0, subsegLen - delta1t.shape[0]),(0,0)), 'constant', constant_values = 0)
                delta2t = np.pad(delta2t, ((0, subsegLen - delta2t.shape[0]),(0,0)), 'constant', constant_values = 0)

            eachData = np.empty((300, filter_num , 3), dtype = np.float32)
            # Normalize
            eachData[:,:,0] = (melt - melMean)/(melStd+eps)
            eachData[:,:,1] = (delta1t - delta1Mean)/(delta1Std+eps)
            eachData[:,:,2] = (delta2t - delta2Mean)/(delta2Std+eps)
            eachData = np.expand_dims(eachData, axis = 0)

            dataList.append(eachData)

    wholeData = np.vstack(dataList)

    return wholeData, subsegNumPerFile


# In[39]:





# In[ ]:




