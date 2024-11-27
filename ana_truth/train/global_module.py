import random
import json,sys
sys.path.append("modules/")

import ROOT;
from ROOT import TTree;
from array import *
from math import *
import math,sys,os
from array import array
from decimal import Decimal
import numpy
import random
import sys,zipfile,json,math


### This is common for all
with open('config.json') as json_file:
    data = json.load(json_file)
    maxNumber=int(data['maxNumber'])
    maxTypes=int(data['maxTypes'])
    mSize=int(data['mSize'])
print ("maxNumber=",maxNumber," maxTypes=",maxTypes," mSize=",mSize)
mSize=maxTypes*maxNumber+1;

# ATLAS PRL cut
CutOutlier_10PB=-9.10
# CMS energy
CMS=13600.0



# dijet invariant mass
x=1+0*maxNumber+1  # X position
y=1+0*maxNumber    # Y position
mjj=(x,y) #  index of Mjj  matrix ellement

# PT of first jet
x=1+0*maxNumber  # X position
y=1+0*maxNumber  # Y position
pT=(x,y) #  index of Mjj  matrix ellement

#  bb mass
x=1+1*maxNumber+1
y=1+1*maxNumber
mbb=(x,y)

#  bj mass
x=1+1*maxNumber
y=1+0*maxNumber
mbj=(x,y)

# mu+mu
x=1+2*maxNumber+1
y=1+2*maxNumber
mmumu=(x,y)

# e+e
x=1+3*maxNumber+1
y=1+3*maxNumber
mee=(x,y)

# j+mu
x=1+2*maxNumber
y=1+0*maxNumber
mjmu=(x,y)

# j+e
x=1+3*maxNumber
y=1+0*maxNumber
mje=(x,y)

# j+gamma
x=1+4*maxNumber
y=1+0*maxNumber
mjg=(x,y)

# b+mu
x=1+2*maxNumber
y=1+1*maxNumber
mbmu=(x,y)

# b+e
x=1+3*maxNumber
y=1+1*maxNumber
mbe=(x,y)

# b+gamma
x=1+4*maxNumber
y=1+1*maxNumber
mbg=(x,y)


############# end invariant mass definitions using RMM ############

### This list contains excluded values for Z-score calculation
### We excluding pT of leading jet, Mjj and mbb
# excluded_val= ( pT, mjj, mbb)
# excluded_val= (mjj, mbb)
# print ("Excluded cells=",excluded_val )

#### Exclusion values for RMM matrix #############
###################################


# dijet invariant mass
x=2 # X position
y=1 # Y position
inx1=x*mSize+y; #  index of hxw matrix ellement

# pT1
x=1 # X position
y=1 # Y position
inx2=x*mSize+y; #  index of hxw matrix ellement

# Mjj for for light-jet + b-jets
x=1+maxNumber # X position
y=1 # Y position
inx3=x*mSize+y; #  index of hxw matrix ellement


# pT1 for for b-jets
x=1+maxNumber # X position
y=1+maxNumber # Y position
inx4=x*mSize+y; #  index # pT for for b-jets

# Mjj for for 2-b jets
x=2+maxNumber # X position
y=1+maxNumber # Y position
inx5=x*mSize+y; #  index of hxw matrix ellement


# exlusion matrix for RMM in terms of indexes (how it is packed)
excluded=(inx1,inx2,inx3,inx4,inx5)



mjjBinsL = [35,44,54,64,75,87,99,112,125,138,151,164,177,190, 203, 216, 229, 243, 257, 272, 287, 303, 319, 335, 352, 369, 387, 405, 424, 443, 462, 482, 502, 523, 544, 566, 588, 611, 634, 657, 681, 705, 730, 755, 781, 807, 834, 861, 889, 917, 946, 976, 1006, 1037, 1068, 1100, 1133, 1166, 1200, 1234, 1269, 1305, 1341, 1378, 1416, 1454, 1493, 1533, 1573, 1614, 1656, 1698, 1741, 1785, 1830, 1875, 1921, 1968, 2016, 2065, 2114, 2164, 2215, 2267, 2320, 2374, 2429, 2485, 2542, 2600, 2659, 2719, 2780, 2842, 2905, 2969, 3034, 3100, 3167, 3235, 3305, 3376, 3448, 3521, 3596, 3672, 3749, 3827, 3907, 3988, 4070, 4154, 4239, 4326, 4414, 4504, 4595, 4688, 4782, 4878, 4975, 5074, 5175, 5277, 5381, 5487, 5595, 5705, 5817, 5931, 6047, 6165, 6285, 6407, 6531, 6658, 6787, 6918, 7052, 7188, 7326, 7467, 7610, 7756, 7904, 8055, 8208, 8364, 8523, 8685, 8850, 9019, 9191, 9366, 9544, 9726, 9911, 10100, 10292, 10488, 10688, 10892, 11100, 11312, 11528, 11748, 11972, 12200, 12432, 12669, 12910, 13156];

mjjBins = array("d", mjjBinsL)



