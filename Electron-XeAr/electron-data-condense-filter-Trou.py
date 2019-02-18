import numpy as np
import glob
import os
from readTRC import readTrc
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chisquare
import numpy.polynomial.polynomial as poly


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def FILE_SORTER(files):
    FilesGold   = []
    FilesSignal = []
    for x in range(0,len(files)):
        if int(files[x][1]) ==3:
            FilesGold.append(files[x])
        if int(files[x][1]) ==2:
            FilesSignal.append(files[x])
            
    FilesGold.sort(); FilesSignal.sort()
    return FilesGold, FilesSignal

def MEAN_WAVEFORM(PATH,files):
    #folders[F]+FilesSignal[0]
    #X, Y, info = readTrc(PATH+files[10])
    #Yvals = np.zeros(X.shape[0])
    #Xvals = np.zeros(Y.shape[0])
    Yvals = np.zeros(250002)
    Xvals = np.zeros(250002)
    avg = 0
    for w in range(0,len(files)):

        X, Y, info = readTrc(PATH+files[w])
        if Y.shape[0]==Yvals.shape[0] and X.shape[0]==Xvals.shape[0]:
            Yvals += Y
            Xvals += X
            avg   += 1
    MSigY = Yvals/avg
    MSigX = Xvals/avg
    return MSigX,MSigY

def WAVEFORM_QUALITY(PATH,files,MEAN,CHI):
    
    #X, Y, info = readTrc(PATH+files[10])
    #Yvals = np.zeros(X.shape[0])
    #Xvals = np.zeros(Y.shape[0])
    Yvals = np.zeros(250002)
    Xvals = np.zeros(250002)
    avg = 0
    #for w in range(0,1100):
    CHII = []
    for w in range(0,len(files)):

        X, Y, info = readTrc(PATH+files[w])
        if Y.shape[0]==Yvals.shape[0] and X.shape[0]==Xvals.shape[0]:
            A = moving_average(MEAN,10)
            B = moving_average(Y   ,10)
            D = moving_average(X   ,10)
            TriggerTime = find_nearest(D,0)
            Trigger     = np.where(D==TriggerTime)[0][0]
            As    = np.mean(A[:Trigger])
            Bs    = np.mean(B[:Trigger])
            
            holder = ((A-As)-(B-Bs))**2
            chi = np.sum(holder)*1e4
            if chi <CHI:
                Yvals += Y
                Xvals += X
                avg   += 1
                CHII.append(chi)
        if avg != 0:
            SigY = Yvals/avg
            SigX = Xvals/avg
        else:
            SigY = Yvals
            SigX = Xvals 
        
    #CHI=np.array(CHI)
    return SigX,SigY,CHII


#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#FOLD = ['100/', '99.9/', '099/', '090/']
#FOLD = [ '099/', '090/']
#FOLD = ['Lamp/']
#FOLD = ['100/']
FOLD = ["090/"]
for per in range(0,len(FOLD)):


    PATH = '/Volumes/MY PASSPORT/ArgonXenon/'+FOLD[per]
    SPATH = '/Users/austinmcdonald/Desktop/ArgonXenon/'+FOLD[per]
    folders = glob.glob(PATH+"*/")
    folders.sort()
    
    CHI = 100
    for F in range(0,len(folders)):
        NAME = folders[F].split('/')[-2]
        print(NAME)
        files = os.listdir(folders[F])
        if '.DS_Store' in files: files.remove('.DS_Store')

        FG,FS = FILE_SORTER(files)
        Mx,MEAN = MEAN_WAVEFORM(folders[F],FG)
        Gx,Gy,ChiG = WAVEFORM_QUALITY(folders[F],FG,MEAN,CHI)

        Mx,MEAN = MEAN_WAVEFORM(folders[F],FS)
        Sx,Sy,ChiS = WAVEFORM_QUALITY(folders[F],FS,MEAN,CHI)

        TriggerTime  = find_nearest(Sx,0)
        Trigger      = np.where(Sx==TriggerTime)[0][0]
        baselineS    = np.mean(Sy[0:Trigger])
        baselineG    = np.mean(Gy[0:Trigger])
        data = [Gx, Gy-baselineG, Sy-baselineS]
        np.savetxt(SPATH+NAME+'.txt',data)


