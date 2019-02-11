import numpy as np
import glob
import os
from definition import *



PATH = '/Users/austinmcdonald/Desktop/ArgonXenon/'

folders = glob.glob(PATH+"*/")
folders.sort()
folders.remove('/Users/austinmcdonald/Desktop/ArgonXenon/Lamp/')
#folders = [folders[1]]
#folders = [folders[1]]
#folders



# arrival, sigma, max, RC
DATA = []
for F in range(0,len(folders)):
   
    files = os.listdir(folders[F])
    if '.DS_Store' in files: files.remove('.DS_Store')
    info = np.loadtxt(folders[F]+'Run-info.txt')
    files.remove('Run-info.txt')
    files.sort()
    N = 0
    for fi in files:
        print("starting on")
        print(fi)
        Data = np.loadtxt(folders[F]+fi)
        if np.mean(np.nan_to_num(Data[0]))!=0 and np.mean(np.nan_to_num(Data[1]))!=0:
            xa,ya,opta,chia,cuta = FITTER_ANOD(Data[0],Data[2])
            xg,yg,optg,chig,cutg = FITTER_GOLD(Data[0],Data[1])
            
            EE =int(fi.split('.txt')[0].split('-')[-1])
            PP = int(fi.split('.txt')[0].split('-')[2].split('_')[0])
            ident = None
            if PP==1:
                PP = info[0]
                ident = '^'
            elif PP==3:
                PP = info[1]
                ident = 'd'
            elif PP==6:
                PP = info[2]
                ident = 's'
            elif PP==9:
                PP = info[3]
                ident = '*'
                
            Prct = folders[F].split('/')[-2]
            dt = opta[0]-optg[0]
            sigma = opta[1]
            #INFO = [Prct, EE, PP, dt, sigma, ident, opta, optg]
            INFO = [float(Prct), EE, PP, dt, sigma, ident]
            #INFO = np.array([int(Prct), EE, PP, dt, sigma, ident])
            
            DATA.append(INFO)
DATA = np.array(DATA)
#print(DATA[:,0])
np.save('XenonArgon.npy',DATA)
