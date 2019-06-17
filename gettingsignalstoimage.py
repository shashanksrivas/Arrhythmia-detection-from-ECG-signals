import cv2
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import csv
#from scipy.signal import find_peaks
import scipy.signal
import peakutils
import string

data = []
f = open("demofile2.txt", "a+")
with open('/home/sas/PycharmProjects/keras-flask-deploy-webapp/mitbih_test.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        data.append(float(row[0]))
        f.write(row[0])
        f.write("\n")
print(data)

#data = np.array('/home/sas/PycharmProjects/sample/ECG-Arrhythmia-classification/ss.csv')
signals = []
count=1
data1=[]
#data1 = [float(x) for x in data]
for item in data:
	data1.append(item)
#print(data1)
#data1=list(map(float, data))
#signal=data
#data=data.tolist();
#peaks = peakutils.indexes(data, thres=0.02/max(data), min_dist=0.1)
#peaks = scipy.signal.argrelextrema(np.array(data),comparator=np.greater,order=2)
#peaks = scipy.signal.find_peaks_cwt(data, np.arange(1, 4),max_distances=np.arange(1, 4)*2)
#peaks = scipy.signal.find_peaks_cwt(data,widths=10)
#peaks = detect_peaks.detect_peaks(data, mph=7, mpd=2)
#peaks, _ = scipy.signal.find_peaks(data, distance=200)
#peaks = peakutils.indexes(data, thres=(0.0002), min_dist=0.1)
#peaks, _ = find_peaks(data, distance=150)
peaks = biosppy.signals.ecg.christov_segmenter(signal = data1, sampling_rate = 100)[0] 
print(peaks)
for i in ( peaks [ 1 : - 1 ]): 
	diff1 = abs ( peaks [ count - 1 ] - i ) 
	diff2 = abs ( peaks [ count + 1 ] - i ) 
	x = peaks [ count - 1 ] + diff1 // 2 
	y = peaks [ count + 1 ] - diff2 // 2
	signal = data [ x : y ] 
	signals . append ( signal ) 
	count += 1
print(signals)
for count, i in enumerate(signals):
  fig = plt.figure(frameon=False)
  plt.plot(i)
  plt.xticks([]), plt.yticks([])
  for spine in plt.gca().spines.values():
     spine.set_visible(False)
  filename = '/home/sas/PycharmProjects/keras-flask-deploy-webapp/' + str(count)+'.png'
  fig.savefig(filename)
  im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
  cv2.imwrite(filename, im_gray)









 
#from biosppy.signals import ecg import biosppy signals ,
#fields = wfdb . rdsamp ( get_records ()[ 0 ], channels = [ 0 ]) 
#display ( signals ) display ( fields ) out = ecg . ecg ( signal = signals . flatten (), sampling_rate = 1000. , show = True ) 
#data = signals . flatten () signals = [] count = 1 
#peaks = biosppy . signals . ecg . christov_segmenter ( signal = data , sampling_rate = 200 )[ 0 ] 
#for i in ( peaks [ 1 : - 1 ]): diff1 = abs ( peaks [ count - 1 ] - i ) diff2 = abs ( peaks [ count + 1 ] - i ) x = peaks [ count - 1 ] + diff1 // 2 y = peaks [ count + 1 ] - diff2 // 2 signal = data [ x : y ] signals . append ( signal ) count += 1
 
