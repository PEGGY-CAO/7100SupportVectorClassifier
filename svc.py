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
import itertools
from sklearn.svm import LinearSVC

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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
allvalences = []

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
        
        # construct the centeredData with subtraction of the average, normalized and bandpassfilter
        lowcut = 8
        highcut = 40
        fs = 128
        
        centeredData = []
        for i in range(40):
            eachTrial = []
            for j in range(32):
#                print max(dataArray[i][j])
                preproc = (dataArray[i][j][baselineIndices:] - baselinePart[i][j])/max(dataArray[i][j][baselineIndices:])
                
                y = butter_bandpass_filter(preproc, lowcut, highcut, fs, order=6)
                eachTrial.append(y)
            centeredData.append(eachTrial)
        # print np.array(centeredData).shape   
        # now centeredData's shape: 40*32*7680
            
        # processing label, label acctually consists of valence, arousal, dominance, liking
        newLable = array["labels"]
        # get valence
        valences = [newLable[i][0] for i in range(40)]
        allvalences.append(np.around(valences))
        
        
        happy =[]
        sad = []
        
        #select 8 matching channels: Fp1, Fp2, O1, O2, T7, T8, P3, P4, P7, P8
        # channelSelect = [0, 16, 13, 31, 7, 25, 10, 28, 11, 29]
        # Fp1, Fp2
        channelSelect = [13, 31, 10, 28, 7, 25, 11, 29]
        
        #select motions with happy and unhappy
        for trial in range(40):
            
            label = valences[trial]
            
            if (label >= 6 and label <= 9) or (label >=1 and label <=3.5): 
                selectedChannels = [centeredData[trial][i] for i in channelSelect]
#                print np.array(selectedChannels).shape
                spec=[]
                
                #scipy spectrogram F-t
                for sig in selectedChannels:
                    
                    f1, t1, cwtmatr1 = signal.spectrogram(sig, 128)
#                    print f1.shape, t1
#                    widths = np.arange(1, 31)
                    alpha = ((f1 >= 8) & (f1 <= 40))
#                    cwtmatr1 = signal.cwt(sig, signal.ricker, widths)
#                    print np.array(Sxx1).shape: 129*34
                    #I want to select the bandwith in range 13-30Hz
                    beta = ((f1 >= 13) & (f1 <= 30))
                    gamma = ((f1 >= 30) & (f1 <= 100))
                    spec.append(cwtmatr1[alpha])
                    
                #    
                if random.uniform(0, 1) < 0.2:
                    x_test.append(((spec[0]+spec[2]-spec[1]-spec[3]+spec[4]-spec[5]+spec[6]-spec[7])/4).flatten())
#                    x_test.append(spec)
                    if (label >= 6 and label <= 9):
                        y_test.append(1)
                    else:
                        y_test.append(0)
                else:
                    x_train.append(((spec[0]+spec[2]-spec[1]-spec[3]+spec[4]-spec[5]+spec[6]-spec[7])/4).flatten())
#                    x_train.append(spec)
                    if (label >= 6 and label <= 9):
                        y_train.append(1)
                    else:
                        y_train.append(0)
                
                
                
#                title = 'Trial: '+str(trial+1)
                
#                fig, axs = plt.subplots(2, 1, sharex='col')
#                fig.suptitle(title)
#                
#                axs[0].pcolormesh(t1, f1, Sxx1)
#                axs[0].set_xlabel('Time [sec]')
#                axs[0].set_ylabel('Frequency [Hz]')                   
#                axs[0].set_title("O1")
#                
#                
#                axs[1].pcolormesh(t2, f2, Sxx2)
#                axs[1].set_ylabel('Frequency [Hz]')
#                axs[1].set_xlabel('Time [sec]')
#                axs[1].set_title("O2")
                
                
#                plt.show()
          
    #                for i in range(nums_chunk):
    #                    #calculate psd for each chunk
    #                    F1, PSDfp1 = signal.welch(selectedChannels[0][i*chunk : (i+1)*chunk], fs=128)
    #                    F1, PSDfp2 = signal.welch(selectedChannels[1][i*chunk : (i+1)*chunk], fs=128)
    #                    
    #                    alpha = ((F1 >= 8) & (F1 <= 12))
    ##                    if trial == 1:
    ##                        plt.figure(i)
    ##                        plt.plot(F1[alpha], PSDfp1[alpha], 'r', label="Fp1")
    ##                        plt.plot(F1[alpha], PSDfp2[alpha], 'b', label="Fp2")
    ##                        plt.show()
    #                        
    #                    # calculate the difference psd between the time interval with index i
    #                    diffFp = np.dot(F1[alpha], PSDfp1[alpha]) - np.dot(F1[alpha], PSDfp2[alpha])
    ##                    psdDiff.append(diffFp)
    ##                average = np.average(psdDiff)
    #                    newData.append(diffFp)
    #                    emotion.append("Happy")
        #                print np.array(PSD2[alpha]).shape
                    
#            if valences[trial] >=1 and valences[trial] <=3.5:
#                selectedChannels = [centeredData[trial][i] for i in channelSelect]
#                
#                #continuous wavelet transform
##                    plt.figure()
#                fig, axs = plt.subplots(2, 1, sharex='col')
#                widths = np.arange(1, 31)
#                
#                cwtmatr1 = signal.cwt(selectedChannels[0], signal.ricker, widths)
#                print cwtmatr1.shape
#                axs[0].imshow(cwtmatr1, extent=[-1,7680,1,40],cmap='PRGn', aspect='auto', vmax=abs(cwtmatr1).max(), vmin=-abs(cwtmatr1).max())
#                cwtmatr2 = signal.cwt(selectedChannels[1], signal.ricker, widths)
#                axs[1].imshow(cwtmatr2, extent=[-1,7680,1,40],cmap='PRGn', aspect='auto', vmax=abs(cwtmatr2).max(), vmin=-abs(cwtmatr2).max())
#                plt.show()
                    
                    
                    
                    
                    
#                    psdDiff = []
#          
#                    for i in range(nums_chunk):
#                        #calculate psd for each chunk
#                        F1, PSDfp1 = signal.welch(selectedChannels[0][i*chunk : (i+1)*chunk], fs=128)
#                        F1, PSDfp2 = signal.welch(selectedChannels[1][i*chunk : (i+1)*chunk], fs=128)
#    
#                        alpha = ((F1 >= 8) & (F1 <= 12))
#    #                    if trial == 1:
#    #                        plt.figure(i)
#    #                        plt.plot(F1[alpha], PSDfp1[alpha], 'r', label="Fp1")
#    #                        plt.plot(F1[alpha], PSDfp2[alpha], 'b', label="Fp2")
#    #                        plt.show()
#                            
#                        # calculate the difference psd between the time interval with index i
#                        diffFp = np.dot(F1[alpha], PSDfp1[alpha]) - np.dot(F1[alpha], PSDfp2[alpha])
#                        newData.append(diffFp)
#    #                average = np.average(psdDiff)
#    #                newData.append(average)
#                        emotion.append("Sad")
                
                
        
#                F1, PSDO1 = signal.welch(selectedChannels[2], fs=128)
#                F1, PSDO2 = signal.welch(selectedChannels[3], fs=128)
#                diffO = np.dot(F1[alpha], PSDO1[alpha]) - np.dot(F1[alpha], PSDO2[alpha])
#                F1, PSDT7 = signal.welch(selectedChannels[4], fs=128)
#                F1, PSDT8 = signal.welch(selectedChannels[5], fs=128)
#                diffT = np.dot(F1[alpha], PSDT7[alpha]) - np.dot(F1[alpha], PSDT8[alpha])
#                F1, PSDP3 = signal.welch(selectedChannels[6], fs=128)
#                F1, PSDP4 = signal.welch(selectedChannels[7], fs=128)
#                diffP = np.dot(F1[alpha], PSDP3[alpha]) - np.dot(F1[alpha], PSDP4[alpha])
#                average = np.average([diffFp, diffO, diffT, diffP])
#                sad.append(average)
print np.array(x_train).shape 
cmatr = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

      
clf = LinearSVC(random_state=0, tol=1e-5,C=0.1,max_iter=200)
clf.fit(x_train, y_train)
#
dataaaa = x_test[0]
label_t = y_test[0]
#
print "acturally", y_test
#print clf.predict(x_test)
print clf.score(x_test, y_test)
#print np.array(allvalences).shape
#allvalences = list(itertools.chain.from_iterable(allvalences))  
#                 
#plt.figure()
#plt.hist(allvalences, bins=np.arange(11), density=True)
#plt.xticks(np.arange(1, 10, 1))
#plt.title('Distribution of Valence in range 1 to 9')
#plt.show()
#
#
##construct panda data framework
#df = pd.DataFrame({"PSD-Asymetric": np.array(newData),
#                   "emotion": emotion})
#print np.array(newData).shape
#print np.array(emotion).shape

#plt.figure()
##draw box and whisker
#sns.set(style="whitegrid")
#
#ax = sns.boxplot(x="emotion", y="PSD-Asymetric", data=df)   
#plt.show()






        
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

