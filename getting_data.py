import cv2
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import csv
#from scipy.signal import find_peaks
import scipy.signal
#import peakutils
import string
from glob import *
import wfdb

paths = glob('/home/dell/ecg/mitbih/*.atr')
# Get rid of the extension
paths = [path[:-4] for path in paths]
paths.sort()
Normal = []
for e in paths:
    signals, fields = wfdb.rdsamp(e, channels = [0]) 
    ann = wfdb.rdann(e, 'atr')
    good = ['N']
    ids = np.in1d(ann.symbol, good)
    imp_beats = ann.sample[ids]
    beats = (ann.sample)
    for i in imp_beats:
        beats = list(beats)
        j = beats.index(i)
        if(j!=0 and j!=(len(beats)-1)):
            x = beats[j-1]
            y = beats[j+1]
            diff1 = abs(x - beats[j])//2
            diff2 = abs(y - beats[j])//2
            Normal.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
#print(Normal)

signals1=[]
for item in signals:
    for item2 in item:
        signals1.append(item2)
print(signals1)
count=1
peaks = biosppy.signals.ecg.christov_segmenter(signal = signals1, sampling_rate = 100)[0] 
#print(peaks)
signalss=[]
for i in ( peaks [ 1 : - 1 ]): 
	diff1 = abs ( peaks [ count - 1 ] - i ) 
	diff2 = abs ( peaks [ count + 1 ] - i ) 
	x = peaks [ count - 1 ] + diff1 // 2 
	y = peaks [ count + 1 ] - diff2 // 2
	signal = signals1 [ x : y ] 
	signalss . append ( signal ) 
	count += 1
print(signalss)

for count, i in enumerate(signalss):
    fig = plt.figure(frameon=False)
    plt.plot(i)
    plt.xticks([]), plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    filename = '/home/dell/ecg/' + str(count)+'.png'
    fig.savefig(filename)
    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
    cv2.imwrite(filename, im_gray)
    if count== 1200:
       exit()

