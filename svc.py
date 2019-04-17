import os
import cPickle
import random
import numpy as np
# from sklearn import svm
import operator
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from scipy import signal
import seaborn as sns
import pandas as pd


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


newData = []
emotion = []
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
        
        # calculate average of the first 3 seconds then get baselinePart(40trial*32channel)
        baselineIndices = 128 * 3
        baselinePart = [np.array(dataArray[i][:, :baselineIndices]).mean(axis=1) for i in range(40)]
        
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
        happy =[]
        sad = []
        
        #select 8 matching channels: Fp1, Fp2, O1, O2, T7, T8, P3, P4
        channelSelect = [0, 16, 13, 31, 7, 25, 10, 28]
        # O1, O2, T7, T8
#        channelSelect = [13, 31, 7, 25]
        
        #normalize Valence from 1-9 to a binary scale 0-1
        for trial in range(40):
            if valences[trial] >= 6 and valences[trial] <= 9:
                eightChannels = [centeredData[trial][i] for i in channelSelect]
            
                F1, PSDfp1 = signal.welch(eightChannels[0], fs=128)
                F1, PSDfp2 = signal.welch(eightChannels[1], fs=128)
                alpha = ((F1 >= 8) & (F1 <= 12))
                if trial == 1:
                    plt.figure(2)
                    plt.plot(F1[alpha], PSDfp1[alpha], 'r', label="Fp1")
                    plt.show()
                
#                print np.array(PSD2[alpha]).shape
                diffFp = np.dot(F1[alpha], PSDfp1[alpha]) - np.dot(F1[alpha], PSDfp2[alpha])
                F1, PSDO1 = signal.welch(eightChannels[2], fs=128)
                F1, PSDO2 = signal.welch(eightChannels[3], fs=128)
                diffO = np.dot(F1[alpha], PSDO1[alpha]) - np.dot(F1[alpha], PSDO2[alpha])
                F1, PSDT7 = signal.welch(eightChannels[4], fs=128)
                F1, PSDT8 = signal.welch(eightChannels[5], fs=128)
                diffT = np.dot(F1[alpha], PSDT7[alpha]) - np.dot(F1[alpha], PSDT8[alpha])
                F1, PSDP3 = signal.welch(eightChannels[6], fs=128)
                F1, PSDP4 = signal.welch(eightChannels[7], fs=128)
                diffP = np.dot(F1[alpha], PSDP3[alpha]) - np.dot(F1[alpha], PSDP4[alpha])
                average = np.average([diffFp, diffO, diffT, diffP])
                happy.append(average)
#                print "DIFFFP ", diffFp
#                print "diffO ", diffO
#                print "diffT: ", diffT
#                print "diffP: ", diffP
                
            if valences[trial] >=1 and valences[trial] <=3:
                eightChannels = [centeredData[trial][i] for i in channelSelect]
                eightChannels = [centeredData[trial][i] for i in channelSelect]
                
                F1, PSDfp1 = signal.welch(eightChannels[0], fs=128)
                F1, PSDfp2 = signal.welch(eightChannels[1], fs=128)
                alpha = ((F1 >= 8) & (F1 <= 12))
#                print np.array(PSD2[alpha]).shape
                diffFp = np.dot(F1[alpha], PSDfp1[alpha]) - np.dot(F1[alpha], PSDfp2[alpha])
                F1, PSDO1 = signal.welch(eightChannels[2], fs=128)
                F1, PSDO2 = signal.welch(eightChannels[3], fs=128)
                diffO = np.dot(F1[alpha], PSDO1[alpha]) - np.dot(F1[alpha], PSDO2[alpha])
                F1, PSDT7 = signal.welch(eightChannels[4], fs=128)
                F1, PSDT8 = signal.welch(eightChannels[5], fs=128)
                diffT = np.dot(F1[alpha], PSDT7[alpha]) - np.dot(F1[alpha], PSDT8[alpha])
                F1, PSDP3 = signal.welch(eightChannels[6], fs=128)
                F1, PSDP4 = signal.welch(eightChannels[7], fs=128)
                diffP = np.dot(F1[alpha], PSDP3[alpha]) - np.dot(F1[alpha], PSDP4[alpha])
                average = np.average([diffFp, diffO, diffT, diffP])
                sad.append(average)
         
        newData.append(np.mean(happy)) 
        emotion.append("happy")
        newData.append(np.mean(sad))
        emotion.append("sad")

#construct panda data framework
df = pd.DataFrame({"PSD-Asymetric": np.array(newData),
                   "emotion": emotion})


#draw box and whisker
sns.set(style="whitegrid")
 
ax = sns.boxplot(x="emotion", y="PSD-Asymetric", data=df)      
        
#        centeredVectors = zip(list(centeredData), binaryLabel)
#        print np.array(centeredVectors).shape
        
        
        # collect data in order to draw box-and-whisker plot
        
        
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
#trial1 = centeredVectors[0][0][0]
#trial1Fp2 = centeredVectors[0][0][16]
#trial1O1 = centeredVectors[0][0][13]
#trial1O2 = centeredVectors[0][0][31]
#
#numbers = np.array(trial1).shape[0]
#print numbers
## numbers = 7680 in this case
#alpha = 12
#
#fft_vals = np.fft.fft(trial1)
#fft_t1_fp1 = 2.0 * np.abs(fft_vals / numbers)
#fftTrial1Fp2 = 2.0 * np.abs(np.fft.fft(trial1Fp2) / numbers)
#
#
#f = 1.0 / 128
#freq = np.fft.fftfreq(numbers, f)
#
#print np.array(freq).shape
## construct an Alpha mask
#mask = ((freq >= 8) & (freq <= 12))
#
#
#plt.figure(2)
#plt.plot(freq[mask], fft_t1_fp1[mask], 'r', label="Fp1")
#plt.plot(freq[mask], fftTrial1Fp2[mask], 'b', label="Fp2")
#plt.legend()
#plt.title("FFT for first trial of participant 1")
#plt.show()
#
## calculate power spectra
#psFp1 = 2.0 * (np.abs(fft_vals / numbers)**2.0)
#psFp2 = 2.0 * (np.abs(np.fft.fft(trial1Fp2) / numbers)**2.0)
#plt.figure(3)
#plt.plot(freq[mask], psFp1[mask], 'r', label="ps-Fp1")
#plt.plot(freq[mask], psFp2[mask], 'b', label="ps-Fp2")
#plt.legend()
#plt.title("power spectrum for Fp1 and Fp2")
#plt.show()
#
#fft_t1_o1 = 2.0 * np.abs(np.fft.fft(trial1O1) / numbers)
#fft_t1_o2 = 2.0 * np.abs(np.fft.fft(trial1O2) / numbers)
#plt.figure(4)
#plt.plot(freq[mask], fft_t1_o1[mask], 'r', label="fft-O1")
#plt.plot(freq[mask], fft_t1_o2[mask], 'b', label="fft-O2")
#plt.legend()
#plt.title("fft for O1 and O2")
#plt.show()
#
#ps_o1 = 2.0 * (np.abs(np.fft.fft(trial1O1) / numbers)**2.0)
#ps_o2 = 2.0 * (np.abs(np.fft.fft(trial1O2) / numbers)**2.0)
#plt.figure(5)
#plt.plot(freq[mask], ps_o1[mask], 'r', label="ps-o1")
#plt.plot(freq[mask], ps_o2[mask], 'b', label="ps-o2")
#plt.legend()
#plt.title("power spectral density for O1 and O2 with fft^2")
#plt.show()
#
#plt.figure(6)
#psd, freqs = plt.psd(trial1O1, Fs=128)
##print psd, freqs
#plt.psd(trial1O2, Fs=128)
#plt.title("power spectral density for O1 and O2 with matplotlib.pyplot.psd")
#plt.show()
#
#plt.figure(7)
#F1, PSD1 = signal.welch(trial1O1, fs=128)
#F2, PSD2 = signal.welch(trial1O2, fs=128)
#print F1, F2
#print PSD1
#alpha = ((F1 >= 8) & (F1 <= 12))
#plt.plot(F1[alpha], PSD1[alpha], 'r', label="psdO1")
#plt.plot(F1[alpha], PSD2[alpha], 'b', label="psdO2")
#
#print sum([a*b for a,b in zip(PSD1[alpha],F1[alpha])])
#
#print sum(PSD1[alpha])
#print sum(PSD2[alpha])
#plt.title("power spectral density for O1 and O2 with scipy welch")
#plt.legend()
#plt.show()

#print trial1

# change list to np.array
#x_test, y_test, x_train, y_train = np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)

