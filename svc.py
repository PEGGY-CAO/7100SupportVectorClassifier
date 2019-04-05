import os
import cPickle
import random
import numpy as np
# from sklearn import svm
import operator

#first, read in the pre-processed data from a folder containing the DEAP .dat files.
print("Loading EEG files")
validation_fragments = []
validation_truth = []
train_fragments = []
train_truth = []
filenames = [f for f in os.listdir("DEAP_data/")
             if (os.path.isfile("DEAP_data/" + f) and '.dat' in f)]

print("Filenames are ", filenames)

x_test = []
y_test = []
x_train = []
y_train = []

# bootstrapping test data(20%) and train data(80%)
'''
for filename in filenames:
    with open("DEAP_data/" + filename, 'rb') as f:
        print(filename)
        array = cPickle.load(f)
        # print("array is", np.array(array))
        for datum, label in zip(list(array["data"]), list(array["labels"])):
            if random.uniform(0, 1) < .2:
                x_test.append(np.array(datum).flatten())
                y_test.append(label[0])
            else:
                x_train.append(np.array(datum).flatten())
                y_train.append(label[0])
'''
for filename in filenames:
    with open("DEAP_data/" + filename, 'rb') as f:
    
        print(filename)
        # Note: EEG data were downsampled to 128 Hz from 512 Hz already, a bandpass filter from 4.0-45.0 Hz was applied already
        array = cPickle.load(f)
        # array is a dictionary with 'data' and 'label'
        
        dataArray = array["data"]
        # print np.array(dataArray).shape and get (40 * 40 * 8064)
        # dataArray's dimension: 40(trial) * 40(channel) * 8064(=128Hz*63) 
           
        # remaining 32 channels of EEG signal and calculate average of the first 3 seconds
        # 128Hz * 3
        dataArray = [dataArray[i][:32,:] for i in range(40)]
        print(np.array(dataArray).shape)
        # calculate average of the first 3 seconds then get baselinePart(40trial*32channel)
        baselineIndices = 128 * 3
        baselinePart = [np.array(dataArray[i][:, :baselineIndices]).mean(axis=1) for i in range(40)]
        print np.array(baselinePart).shape
        
        #construct the centeredData with subtraction of the average
        centeredData = []
        for i in range(40):
            eachTrial = []
            for j in range(32):
                eachTrial.append(dataArray[i][j][baselineIndices:] - baselinePart[i][j])
            centeredData.append(eachTrial)
        print np.array(centeredData).shape        
            
        
        
        newLable = array["labels"]
            #get Valence
        # print newLable
        newLable = [newLable[i][0] for i in range(40)]
        #each lable are with content(valence, arousal, dominance, liking)
            # print(newLable)
        temp =[]
    #normalize Valence from 1-9 to a binary scale 0-1
        for times in range(40):
            if newLable[times] >= 5:
                temp.append(1)
            else:
                temp.append(0)
        x = zip(list(dataArray), temp)
    #select four channels of EEG as representation
        b = [0, 3, 16, 20]
        for datum, label in x:
            if random.uniform(0,1) < .2:
    # print(datum)
                x_test.append(np.array(operator.itemgetter(*b)(np.array(datum))).flatten())
                y_test.append(label)
            else:
                x_train.append(np.array(operator.itemgetter(*b)(np.array(datum))).flatten())
                y_train.append(label)

print np.array(x_test).shape
#print(np.array(x_train).shape)



# change list to np.array
#x_test, y_test, x_train, y_train = np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)

