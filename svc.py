import os
import cPickle
import random
import numpy as np
# from sklearn import svm
import operator
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft


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
        if filename == 's01.dat':
            wantToCheck = dataArray[0][0]
            plt.figure(1)
            plt.plot(wantToCheck)
            plt.title('original preprocessed data of participant 1 trial 1')
            plt.show()
            print wantToCheck
            print ('average',np.average(wantToCheck))
            print ('median', np.median(wantToCheck))
            print np.array(wantToCheck).shape
       
        # calculate average of the first 3 seconds then get baselinePart(40trial*32channel)
        baselineIndices = 128 * 3
        baselinePart = [np.array(dataArray[i][:, :baselineIndices]).mean(axis=1) for i in range(40)]
#        print baselinePart[0][0]
        
        # construct the centeredData with subtraction of the average
        centeredData = []
        for i in range(40):
            eachTrial = []
            for j in range(32):
                eachTrial.append(dataArray[i][j][baselineIndices:] - baselinePart[i][j])
            centeredData.append(eachTrial)
        # print np.array(centeredData).shape   
        # now centeredData's shape: 40*32*7680
            
        # processing label, label acctually consists of valence, arousal, dominance, liking
        newLable = array["labels"]
        # get valence
        valences = [newLable[i][0] for i in range(40)]
        binaryLabel =[]
        #normalize Valence from 1-9 to a binary scale 0-1
        for times in range(40):
            if valences[times] >= 5:
                binaryLabel.append(1)
            else:
                binaryLabel.append(0)
        centeredVectors = zip(list(centeredData), binaryLabel)
#        print np.array(centeredVectors).shape
        #select six channels: Fp1, Fp2, O1, O2, T8, P4
        channelSelect = [0, 16, 13, 31, 25, 29]
        
#        for datum, label in centeredVectors:
#            if random.uniform(0,1) < .2:
#                x_test.append(np.array(operator.itemgetter(*b)(np.array(datum))).flatten())
#                y_test.append(label)
#            else:
#                x_train.append(np.array(operator.itemgetter(*b)(np.array(datum))).flatten())
#                y_train.append(label)

#print np.array(x_test).shape
#print(np.array(x_train).shape) 
        
# I want to plot FFT of the first participant's first trial of channel Fp1
trial1 = centeredVectors[0][0][0]
trial1Fp2 = centeredVectors[0][0][16]

numbers = np.array(trial1).shape[0]
print numbers
# numbers = 7680 in this case
alpha = 12

fft_vals = np.fft.fft(trial1)
fft_theo = 2.0 * np.abs(fft_vals / numbers)
fftTrial1Fp2 = 2.0 * np.abs(np.fft.fft(trial1Fp2) / numbers)
f = 1.0 / 128
freq = np.fft.fftfreq(numbers, f)

print np.array(freq).shape
mask = freq > 0
plt.figure(2)
plt.plot(freq[mask], fft_theo[mask], 'r', label="Fp1")
plt.plot(freq[mask], fftTrial1Fp2[mask], 'b', label="Fp2")
plt.legend()
plt.title("FFT for first trial of participant 1")
plt.show()

# calculate power spectra
psFp1 = 2.0 * (np.abs(fft_vals / numbers)**2.0)
psFp2 = 2.0 * (np.abs(np.fft.fft(trial1Fp2) / numbers)**2.0)
plt.figure(3)
plt.plot(freq[mask], psFp1[mask], 'r', label="ps-Fp1")
plt.plot(freq[mask], psFp2[mask], 'b', label="ps-Fp2")
plt.legend()
plt.title("power spectrum for Fp1 and Fp2")
plt.show()

#print trial1

# change list to np.array
#x_test, y_test, x_train, y_train = np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)

