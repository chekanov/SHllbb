import random
import sys
sys.path.append("modules/")

# import atlas styles
from array import *
from math import *
from ROOT import TH1D,TF1,TCanvas,TColor,TPostScript,TProfile2D,THStack,TRandom3,TFile,TLatex,TLegend,TPaveText,TGraphErrors,kRed,kBlue,kGreen,kCyan,kAzure,kYellow,kTRUE
import math,sys,os
import shapiro
import ROOT
from array import array
from decimal import Decimal
import sys,zipfile,json,math
from ROOT import gROOT, gPad, gStyle, gRandom
import math,sys,os 
import numpy
import random
import sys,zipfile,json,math
from array import *
from math import *

CMS=13000.0

# Run 2 and Run3 in TeV
CMS_RUN2=CMS
CMS_RUN3=13600.0

# Run2 and Run3 in TeV
CMS_RUN2_TEV=CMS_RUN2*0.001 
CMS_RUN3_TEV=CMS_RUN3*0.001

# minimum Run for Run2. After this run number, run3 starts
RUN3_MIN_RUN=420000


## masses labels 
masseslab=["jj","jb","bb","je","j\;\mu","j\;\gamma","b\;\ell","b\;\mu","b\;\gamma"]
## masses data in histograms
massesdata=["jj","jb","bb","je","jm","jg","be","bm","bg"]


# triggers
labels=["MET", "\ell", "2 \ell ", "1 #gamma ", "2 #gamma ", "1j ", "4j "]
labels_map={1:"MET",2:"l",3:"2 l",4:"1 #gamma ",5:"2 #gamma ",6:"1j ",7:"4j "}
labels_color=[1, 2, 4, 6, 8, 44, 1]
labels_style=[24, 20, 21, 22, 23, 20, 34]

xmapcolor={}
xmapcolor[500]=31
xmapcolor[700]=32
xmapcolor[1000]=33
xmapcolor[1500]=34
xmapcolor[2000]=35


#########################################################
# cut to select outlier events

# exopected lumin in pb
ExpectedLumiFB=140


# 20 pb WP
CutOutlier_20PB=-9.39

# MC region for 10 pb working point ("data limit") 
CutOutlier_10PB=-9.10 

# WP region for 1 pb working point 
CutOutlier_1PB=-8.0

# WP region 0.1 pb
CutOutlier_01PB=-6.50 


# main cuts
CutOutlierData=CutOutlier_01PB
CutOutlierMC=CutOutlier_10PB

DataLab="Data #sqrt{s}=13 TeV"
KinemCuts="E_{T}^{jet}>410 GeV  |#eta^{#gamma}|<2.5";
ATLASprel="ATLAS internal"
mcTT="PYTHIA t#bar{t}+single t"
mcWZ="PYTHIA W/Z+jet"


mjjBinsL = [0, 14,28,43,58,72,86,99,112,125,138,151,164,177,190, 203, 216, 229, 243, 257, 272, 287, 303, 319, 335, 352, 369, 387, 405, 424, 443, 462, 482, 502, 523, 544, 566, 588, 611, 634, 657, 681, 705, 730, 755, 781, 807, 834, 861, 889, 917, 946, 976, 1006, 1037, 1068, 1100, 1133, 1166, 1200, 1234, 1269, 1305, 1341, 1378, 1416, 1454, 1493, 1533, 1573, 1614, 1656, 1698, 1741, 1785, 1830, 1875, 1921, 1968, 2016, 2065, 2114, 2164, 2215, 2267, 2320, 2374, 2429, 2485, 2542, 2600, 2659, 2719, 2780, 2842, 2905, 2969, 3034, 3100, 3167, 3235, 3305, 3376, 3448, 3521, 3596, 3672, 3749, 3827, 3907, 3988, 4070, 4154, 4239, 4326, 4414, 4504, 4595, 4688, 4782, 4878, 4975, 5074, 5175, 5277, 5381, 5487, 5595, 5705, 5817, 5931, 6047, 6165, 6285, 6407, 6531, 6658, 6787, 6918, 7052, 7188, 7326, 7467, 7610, 7756, 7904, 8055, 8208, 8364, 8523, 8685, 8850, 9019, 9191, 9366, 9544, 9726, 9911, 10100, 10292, 10488, 10688, 10892, 11100, 11312, 11528, 11748, 11972, 12200, 12432, 12669, 12910, 13156];

mjjBins = array("d", mjjBinsL)


from os.path import exists

confile='data/config.json'
file_exists = exists( confile )

if (file_exists):
 with open( confile ) as json_file:
    data = json.load(json_file)
    maxNumber=int(data['maxNumber'])
    maxTypes=int(data['maxTypes'])
    mSize=int(data['mSize'])
    print("Read from file ",confile)
else:
    maxNumber=10
    maxTypes=5
    mSize=5


print ("maxNumber=",maxNumber," maxTypes=",maxTypes," mSize=",mSize) 
mSize=maxTypes*maxNumber+1;

######################### define position of invarinat masses from RMM #################

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


# get a labale for trigger 
def getTriggerLabel(trig_type):
  Tlab="1 lep"
  # the algoritm  the algorithm: HighsPt cut x 3 
  if (int(trig_type)==1):
             Tlab="T1:\; MET"
  if (int(trig_type)==2): # 1 lepton 
             Tlab="T2:\; 1 \ell"
  if (int(trig_type)==3): # 2 lepton pT>25 GeV  
             Tlab="T3:\; 2 \ell"
  if (int(trig_type)==4): #  single photon 
             Tlab="T4:\; 1 \gamma"
  if (int(trig_type)==5): #  2 photon    
             Tlab="T5:\; 2 \gamma"
  if (int(trig_type)==6): #  single jet (pT>500 GeV)  
             Tlab="T6:\; 1 jet"
  if (int(trig_type)==7): #  4 jet  (lead 200 GeV) 
             Tlab="T7:\; 4 jets"
  return Tlab



# Save mathplot in CSV file
import csv
def SavePlotXY(xfile,lines, Xlab="X", Ylab="Y"):
   #NrLines=len(lines)
   #print("Nr of lines",NrLines)
   print("Save plot in CSV ",xfile);
   with open(xfile, 'w') as myfile:
            data=lines[0].get_data()
            writer = csv.writer(myfile)
            writer.writerow([Xlab, Ylab])
            for i in range(len(data[0])):
                writer.writerow([data[0][i], data[1][i]])

from numpy import savetxt
def SaveNumpyData(xfile,lines):
    savetxt(xfile,lines,delimiter=',')

# do counting statistics for >3 events
# for less, increase the error 
# Set to 0 if Nr of entries less than 1 
def countingErrors(hhh):
  for i in range(1, hhh.GetNbinsX()):
     D = hhh.GetBinContent(i);
     if (D>3.0): 
              hhh.SetBinError(i, sqrt(D) )
     if (D>0.999 and D<=3): 
              hhh.SetBinError(i, 2*sqrt(D) )
     if (D<1.0):
              hhh.SetBinError(i,0)
              hhh.SetBinContent(i,0)


# calculate loss cut for 10% and 1 % of data
def findCutvalues(hhh):
  for i in range(40, 150, 1):
     cutval = i/10
     intigral = hhh.Integral(hhh.FindBin(cutval), hhh.FindBin(0))
     ratio = intigral/hhh.Integral()
     if (ratio == 0.1 or (math.isclose(ratio, 0.1, abs_tol = 0.01)==1)):
        Xcut1=cutval
     elif (ratio == 0.01 or (math.isclose(ratio, 0.01, abs_tol = 0.01)==1)):
        Xcut2=cutval
  return [Xcut1,Xcut2]
    

# do counting statistics only if errors are smaller than counting 
def countingErrorsCorrect(hhh):
  for i in range(1, hhh.GetNbinsX()):
     D = hhh.GetBinContent(i);
     E=  hhh.GetBinError(i);
     Expected= 0;
     if (D>1):  Expected= sqrt(D)
     if (E< Expected and D>1):
              hhh.SetBinError(i, sqrt(D) )


def SavePlotHisto(xfile,ax):
   print("Save histogram in CSV ",xfile);
   p = ax.patches
   with open(xfile, 'w') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(["Xlow", "Height"])
            for i in range(len(p) ):
                lower_left_corner=p[i].get_xy() 
                #writer.writerow([ lower_left_corner[0], p[i].get_width(), p[i].get_height()  ])
                #writer.writerow([ lower_left_corner[0], p[i].get_height()  ])
                writer.writerow([ lower_left_corner[0], lower_left_corner[1]  ])
## draw axis
def drawXAxis(sf,gPad,XMIN,YMIN,XMAX,YMAX,nameX,nameY,showXAxis=True, showYAxis=True):
 h=gPad.DrawFrame(XMIN,YMIN,XMAX,YMAX);
 ay=h.GetYaxis();
 ay.SetLabelFont(42)

 if (sf==1):
             ay.SetLabelSize(0.05)
             ay.SetTitleSize(0.06)

 if (sf==2 or sf==3):
             ay.SetLabelSize(0.10)
             ay.SetTitleSize(0.3)
 if (sf==20):
             ay.SetLabelSize(0.18)
 if (sf==30):
             ay.SetLabelSize(0.12)
# ay.SetTitleSize(0.1)
 ay.SetNdivisions(505);
 if (sf==1): ay.SetTitle( nameY )
 # ay.Draw("same")
 ax=h.GetXaxis();
 if (sf==1 or sf==2): ax.SetTitle( nameX );
 if (sf==30): ax.SetTitle( nameX );
 ax.SetTitleOffset(1.18)
 ay.SetTitleOffset(0.8)

 ax.SetLabelFont(42)
 # ax.SetTitleFont(42)
 ay.SetLabelFont(42)
 # ay.SetTitleFont(42)
 ax.SetLabelSize(0.12)
 ax.SetTitleSize(0.14)

 if (showXAxis==False):
         ax.SetLabelSize(0)
         ax.SetTitleSize(0)
 if (showYAxis):
          ay.SetLabelSize(0)
          ay.SetTitleSize(0)

 #ay.SetTitleSize(0.14)
 if (sf==30):
          ax.SetLabelSize(0.12)
          ax.SetTitleSize(0.12)
 if (sf==2 or sf==3):
             ay.SetLabelSize(0.12)
             ay.SetTitleSize(0.2)

 ax.Draw("same");
 ay.Draw("same");
 return h

def style3par(back):
     back.SetNpx(100); back.SetLineColor(4); back.SetLineStyle(1)
     back.SetLineWidth(2)
     back.SetParameter(0,4.61489e-02)
     back.SetParameter(1,1.23190e+01)
     back.SetParameter(2,3.65204e+00)

     #back.SetParameter(3,-6.81801e-01)
     #back.SetParLimits(0,0,100)
     # back.SetParLimits(1,0,12)
     #back.SetParLimits(2,-100,100)
     return back

def style5par(back):
     back.SetNpx(200); back.SetLineColor(4); back.SetLineStyle(1)
     back.SetLineWidth(2)
     back.SetParameter(0,6.0e+10)
     back.SetParameter(1,80)
     back.SetParameter(2,40)
     back.SetParameter(3,11)
     back.SetParameter(4,1.0)
     #back.SetParLimits(0,0,10000)
     #back.SetParLimits(1,0,100000000)
     #back.SetParLimits(2,-10000,10000)
     #back.SetParLimits(3,-400,400)
     return back

# get width of the bin near the mass
def getBinWidth(bins,peak):
    imean = bins.FindBin(peak)
    return bins.GetBinCenter(imean+1) - bins.GetBinCenter(imean);


# get run from MC sample
def getRun(sample):
    parts=sample.split(".")
    run=parts[1]
    return run 

# residual plots: input histoogram, function, file name 
# http://sdittami.altervista.org/shapirotest/ShapiroTest.html
from module_functions import  Gauss
def showResiduals(hh,func,fname, MyMinX=-12,  MyMaxX=12, isKS=True):
   print ("showResiduals: Calculate residuals.")
   MyBins=100
   res=TH1D("Residuals","Residuals",MyBins,MyMinX,MyMaxX);
   res.SetTitle("")
   res.SetStats(1)
   res.SetLineWidth(2)
   res.SetMarkerColor( 1 )
   res.SetMarkerStyle( 20 )
   res.SetMarkerSize( 0.8 )
   res.SetFillColor(42)
   nameX="D_{i} - F_{i} / #Delta D_{i}"
   nameY="Entries"
   FitMin=func.GetXmin()
   FitMax=func.GetXmax()
   print ("Fit min=",FitMin,"  max=",FitMax)
   nres=0.0
   residuals=[]
   for i in range(1,hh.GetNbinsX()):
     center=hh.GetBinCenter(i)
     if (hh.GetBinContent(i)>0 and center>FitMin and center<FitMax):
       center=hh.GetBinCenter(i)
       x=hh.GetBinCenter(i)
       D = hh.GetBinContent(i);
       Derr = hh.GetBinError(i);
       B = func.Eval(center);
       frac=0
       if Derr>0:
          frac = (D-B)/Derr
       residuals.append(frac)
       res.Fill(frac)
       nres=nres+1.0
   res.SetStats(1)
   bcallable=Gauss()
   back=TF1("back",bcallable,MyMinX,MyMaxX,3);
   back.SetNpx(200); back.SetLineColor(4); back.SetLineStyle(1)
   back.SetLineWidth(2)
   back.SetParameter(0,10)
   back.SetParameter(1,0)
   back.SetParameter(2,1.0)
   back.SetParLimits(2,0.1,1000)
   #back.SetParLimits(0,0.01,10000000)
   #back.SetParLimits(1,-5.0,5.0)
   #back.SetParLimits(2,0.0,5.0)

   #back.FixParameter(1,0)
   #back.FixParameter(2,1.0)
   nn=0
   chi2min=10000
   parbest=[]
   for i in range(10):
     fitr=res.Fit(back,"SMR0")
     print ("Status=",int(fitr), " is valid=",fitr.IsValid())
     if (fitr.IsValid()==True):
             chi2=back.GetChisquare()/back.GetNDF()
             if chi2<chi2min:
                    nn=nn+1
                    if nn>3:
                           break;
                    back.SetParameter(0,random.randint(0,10))
                    back.SetParameter(1,random.randint(-1,1))
                    back.SetParameter(2,random.randint(0,2.0))
                    par = back.GetParameters()

   #fitr=res.Fit(back,"SMR0")
   fitr.Print()
   print ("Is valid=",fitr.IsValid())

   par = back.GetParameters()
   err=back.GetParErrors()
   chi2= back.GetChisquare()
   ndf=back.GetNDF()
   print ("Chi2=", chi2," ndf=",ndf, " chi2/ndf=",chi2/ndf)
   prob=fitr.Prob();
   print ("Chi2 Probability=",fitr.Prob());
   # make reference for normal
   norm_mean=0
   norm_width=1
   normal=TH1D("Normal with sig=1","Reference normal",MyBins,MyMinX,MyMaxX);
   normal.SetLineWidth(3)
   normal.SetLineColor(2)
   # normal.SetFillColor( 5 )

   maxe=5000
   r = TRandom3()
   for i in range (maxe) :
          xA = r.Gaus(norm_mean, norm_width)
          normal.Fill(xA)
   norM=nres/maxe
   normal.Scale(norM)
   pKSbinned = res.KolmogorovTest(normal)
   KSprob="KS prob ="+"{0:.2f}".format(pKSbinned)
   if (isKS): print (KSprob)

   shapiro_prob=shapiro.ShapiroWilkW(residuals)
   Shapiro="ShapiroWilk ={0:.2f}".format(shapiro_prob)
   print (Shapiro)

   gROOT.SetStyle("ATLAS");
   gStyle.SetOptStat(220002210);
   gStyle.SetStatW(0.32)
   c2=TCanvas("c","BPRE",10,10,600,540);
   c2.Divide(1,1,0.008,0.007);
   c2.SetBottomMargin(0.1)
   c2.SetTopMargin(0.05)
   c2.SetRightMargin(0.02)
   c2.SetLeftMargin(0.10)

   binmax = normal.GetMaximumBin();
   Ymax=normal.GetBinContent(normal.FindBin(0));
   for i in range(1,res.GetNbinsX()):
      if res.GetBinContent(i)>Ymax: Ymax=res.GetBinContent(i);
   Ymax=1.15*Ymax;

   #h=gPad.DrawFrame(MyMinX,0,MyMaxX,Ymax)

   ps2 = TPostScript( fname,113)

   res.SetStats(1)
   gStyle.SetOptStat(220002200);
   gStyle.SetStatW(0.32)

   res.SetAxisRange(0, Ymax,"y");
   res.Draw("histo")
   back.Draw("same")
   if (isKS): normal.Draw("histo same")
   leg2=TLegend(0.11, 0.6, 0.39, 0.90);
   leg2.SetBorderSize(0);
   leg2.SetTextFont(62);
   leg2.SetFillColor(10);
   leg2.SetTextSize(0.04);
   leg2.AddEntry(res,"Residuals","f")
   leg2.AddEntry(back,"Gauss fit","l")
   mean= "mean="+"{0:.2f}".format(par[1])
   mean_err= "#pm "+"{0:.2f}".format(err[1])
   sig= "#sigma="+"{0:.2f}".format(par[2])
   sig_err= "#pm "+"{0:.2f}".format(err[2])
   leg2.AddEntry(back,mean+mean_err,"")
   leg2.AddEntry(back,sig+sig_err,"")
   leg2.AddEntry(back,"#chi^{2}/ndf="+"{0:.2f}".format(chi2/ndf)+"(p="+"{0:.2f})".format(prob),"")
   leg2.AddEntry(back,Shapiro,"")
   if (isKS): leg2.AddEntry(normal,"Normal (#sigma=1)","l")
   if (isKS): leg2.AddEntry(back,KSprob,"")

   leg2.Draw("same");
   ax=res.GetXaxis();
   ax.SetTitle( nameX );
   ay=res.GetYaxis();
   ay.SetTitle( nameY );
   ax.SetTitleOffset(1.0); ay.SetTitleOffset(1.0)
   ax.Draw("same")
   ay.Draw("same")
   gPad.RedrawAxis()
   c2.Update()
   ps2.Close()
   c2.Close();
   print (fname, " done")

# for any but jetjet sample
def getPrediction(inputroot,histo_name):
      global lumi
      #print("Process=",inputroot)
      rf=TFile(inputroot)
      print("Open=",inputroot)
      nCPU=(rf.Get("cpucores")).GetBinContent(2)
      #sumWeights=( (rf.Get("cutflow_weighted")).GetBinContent(1) ) / nCPU
      # cross section time efficiency in pb 
      m_xsec=( (rf.Get("CrossSection")).GetBinContent(2) )/nCPU 
      # determine a luminosity weight
      # total scaling = (sum of event weights)-1 x (filter efficiency) x (k-factor) x (cross section) x (luminosity)
      # print("CPU=",nCPU," sumOfWeights=",sumWeights," cross [pb]=",m_xsec)
      tmp=rf.Get(histo_name)


      sumW=rf.Get("cutflow")
      #sumW.Print("All")
      sumWeights=(sumW.GetBinContent(1))/nCPU
      print("sumWeights=",sumWeights, " Cross x eff=",m_xsec," Lumi=",lumi)
      #sumWeights=tmp.GetSumOfWeights() 
      #sumWeights2 = tmp.GetSumw2().GetSum();
      # print("New sumWeights=",sumWeights, " w**2", sumWeights2 )
      #LumiWeight = 0;
      #if (sumWeights>0): LumiWeight  = m_xsec / sumWeights;
      #if (sumWeights>0): LumiWeight  = m_xsec*lumi 
      #print("Lumi=",lumi," x_sec=", m_xsec, " sumWeights=",sumWeights);
      LumiWeight  = m_xsec*lumi

      tmp_integral=tmp.Integral()
      print("Integral of histogram=",tmp_integral)
 
      ScalingFactor=LumiWeight/(nCPU * sumWeights) 
      # if (tmp_integral>0): ScalingFactor=tmp_integral/LumiWeight 

      print("Scaling=",ScalingFactor)

      """
      if (inputroot.find("364705")>-1): LumiWeight=LumiWeight*3;
      if (inputroot.find("364706")>-1): LumiWeight=LumiWeight*20;
      if (inputroot.find("364707")>-1): LumiWeight=LumiWeight*50;
      if (inputroot.find("364708")>-1): LumiWeight=LumiWeight*50;
      if (inputroot.find("364709")>-1): LumiWeight=LumiWeight*20;
      """

      tmp=tmp.Clone()
      tmp.SetDirectory(0)
      rf.Close()
      tmp.Scale( ScalingFactor )
      #countingErrors(tmp)
      return tmp 



# get a histogram from input file (simple), but use lumi weigths if needed 
def getHistogram(inputroot,histo_name,LumiWeight=1):
      global lumi
      print(inputroot,histo_name)
      rf=TFile(inputroot)
      #rf.ls()
      tmp=rf.Get(histo_name)
      tmp=tmp.Clone()
      tmp.SetDirectory(0)
      tmp.Scale(LumiWeight)
      rf.Close()
      #tmp.Scale(LumiWeight)
      #countingErrors(tmp)
      return tmp


higgs2=["mc20_13TeV:mc20_13TeV.600465.PhPy8EG_PDF4LHC15_HHbbWW2l_cHHH01d0.deriv.DAOD_PHYSLITE.e8222_s3681_r13145_p5855",
        "mc20_13TeV:mc20_13TeV.600467.PhPy8EG_PDF4LHC15_HHbbtt2l_cHHH01d0.deriv.DAOD_PHYSLITE.e8222_s3681_r13145_p5855", 
        "mc20_13TeV:mc20_13TeV.600469.PhPy8EG_PDF4LHC15_HHbbZZ2l_cHHH01d0.deriv.DAOD_PHYSLITE.e8222_s3681_r13145_p5855" 
]

#higgs2=["mc20_13TeV:mc20_13TeV.600465.PhPy8EG_PDF4LHC15_HHbbWW2l_cHHH01d0.deriv.DAOD_PHYSLITE.e8222_s3681_r13145_p5855",
#        "mc20_13TeV:mc20_13TeV.600469.PhPy8EG_PDF4LHC15_HHbbZZ2l_cHHH01d0.deriv.DAOD_PHYSLITE.e8222_s3681_r13145_p5855"
#]


jetjet=[
#        "mc20_13TeV.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
        "mc20_13TeV.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
#        "mc20_13TeV.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
#        "mc20_13TeV.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631",
#        "mc20_13TeV.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW.deriv.DAOD_PHYS.e7142_s3681_r13145_p5631"
         ]


sttbar=[
        "mc20_13TeV.410472.PhPy8EG_A14_ttbar_hdamp258p75_dil.deriv.DAOD_PHYS.e6348_s3681_r13145_p5631",
        "mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYS.e6337_s3681_r13145_p5631",
        "mc20_13TeV.410471.PhPy8EG_A14_ttbar_hdamp258p75_allhad.deriv.DAOD_PHYS.e6337_s3681_r13145_p5631",
        "mc20_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_PHYS.e6527_s3681_r13145_p5631",
        "mc20_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_PHYS.e6527_s3681_r13145_p5631"
        "mc20_13TeV.410646.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_top.deriv.DAOD_PHYS.e6552_s3681_r13145_p5631",
        "mc20_13TeV.410647.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_antitop.deriv.DAOD_PHYS.e6552_s3681_r13145_p5631",
        "mc20_13TeV.410648.PowhegPythia8EvtGen_A14_Wt_DR_dilepton_top.deriv.DAOD_PHYS.e6615_s3681_r13145_p5631",
        "mc20_13TeV.410658.PhPy8EG_A14_tchan_BW50_lept_top.deriv.DAOD_PHYS.e6671_s3681_r13145_p5631", 
        "mc20_13TeV.410659.PhPy8EG_A14_tchan_BW50_lept_antitop.deriv.DAOD_PHYS.e6671_s3681_r13145_p5631", 
        "mc20_13TeV:mc20_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_PHYSLITE.e5070_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_PHYSLITE.e5070_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_PHYSLITE.e5070_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_PHYSLITE.e5070_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.410219.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttmumu.deriv.DAOD_PHYSLITE.e5070_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.410220.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_tttautau.deriv.DAOD_PHYSLITE.e5070_s3681_r13144_p5855"]


# old wzjets 
"""
swjets=["mc20_13TeV.361100.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusenu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361101.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusmunu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361102.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplustaunu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361103.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wminusenu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361104.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wminusmunu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361105.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wminustaunu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
        "mc20_13TeV.361108.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Ztautau.deriv.DAOD_PHYS.e3601_s3681_r13145_p5631",
]
"""

# new
swjets=["mc20_13TeV:mc20_13TeV.700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8527_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8527_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8527_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700358.Sh_2211_Zee2jets_Min_N_TChannel.deriv.DAOD_PHYSLITE.e8357_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700364.Sh_2211_Wtaunu2jets_Min_N_TChannel.deriv.DAOD_PHYSLITE.e8357_s3681_r13144_p5855",
        "mc20_13TeV.700320.Sh_2211_Zee_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855" 
        "mc20_13TeV:mc20_13TeV.700326.Sh_2211_Ztautau_LL_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700327.Sh_2211_Ztautau_LL_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700328.Sh_2211_Ztautau_LL_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700329.Sh_2211_Ztautau_LH_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700330.Sh_2211_Ztautau_LH_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700331.Sh_2211_Ztautau_LH_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700332.Sh_2211_Ztautau_HH_maxHTpTV2_BFilter.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700333.Sh_2211_Ztautau_HH_maxHTpTV2_CFilterBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855", 
        "mc20_13TeV:mc20_13TeV.700334.Sh_2211_Ztautau_HH_maxHTpTV2_CVetoBVeto.deriv.DAOD_PHYSLITE.e8351_s3681_r13144_p5855" 
]


# dibosons
Sdiboson=["mc20_13TeV.700587.Sh_2212_lllljj.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700588.Sh_2212_lllvjj.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700589.Sh_2212_llvvjj_os.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700600.Sh_2212_llll.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700601.Sh_2212_lllv.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700602.Sh_2212_llvv_os.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700603.Sh_2212_llvv_ss.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700604.Sh_2212_lvvv.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700605.Sh_2212_vvvv.deriv.DAOD_PHYS.e8433_s3681_r13145_p5631",
          "mc20_13TeV.700496.Sh_2211_ZbbZvv.deriv.DAOD_PHYSLITE.e8338_s3681_r13144_p5855",
          "mc20_13TeV.700495.Sh_2211_ZqqZvv.deriv.DAOD_PHYSLITE.e8338_s3681_r13145_p5855",
          "mc20_13TeV.700494.Sh_2211_ZbbZll.deriv.DAOD_PHYSLITE.e8338_s3681_r13144_p5855",
          "mc20_13TeV.700493.Sh_2211_ZqqZll.deriv.DAOD_PHYSLITE.e8338_s3681_r13144_p5855",
          "mc20_13TeV.700492.Sh_2211_WqqZll.deriv.DAOD_PHYSLITE.e8338_s3681_r13144_p5855",
          "mc20_13TeV.700491.Sh_2211_WqqZvv.deriv.DAOD_PHYSLITE.e8338_s3681_r13144_p5855",
          "mc20_13TeV.700490.Sh_2211_WlvZbb.deriv.DAOD_PHYSLITE.e8338_s3681_r13144_p5855"
       ]


# photons 
Sphotons=["mc20_13TeV.364351.Sherpa_224_NNPDF30NNLO_Diphoton_myy_50_90.deriv.DAOD_PHYS.e6452_s3681_r13145_p5631",
          "mc20_13TeV.364352.Sherpa_224_NNPDF30NNLO_Diphoton_myy_90_175.deriv.DAOD_PHYS.e6452_s3681_r13145_p5631",
          "mc20_13TeV.364353.Sherpa_224_NNPDF30NNLO_Diphoton_myy_175_2000.deriv.DAOD_PHYS.e7081_s3681_r13145_p5631",
          "mc20_13TeV.364354.Sherpa_224_NNPDF30NNLO_Diphoton_myy_2000_E_CMS.deriv.DAOD_PHYS.e7081_s3681_r13145_p5631",
          "mc20_13TeV.410389.MadGraphPythia8EvtGen_A14NNPDF23_ttgamma_nonallhadronic.deriv.DAOD_PHYS.e6155_s3681_r13145_p5631",
          "mc20_13TeV.700007.Sh_228_yyy_01NLO.deriv.DAOD_PHYS.e7999_s3681_r13145_p5631",
          "mc20_13TeV.364541.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_17_35.deriv.DAOD_PHYS.e6788_s3681_r13145_p5631",
          "mc20_13TeV.364542.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_35_70.deriv.DAOD_PHYS.e6788_s3681_r13145_p5631",
          "mc20_13TeV.364543.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_70_140.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364544.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_140_280.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364545.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_280_500.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364546.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_500_1000.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364547.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_1000_E_CMS.deriv.DAOD_PHYS.e6068_s3681_r13145_p5631",
          "mc20_13TeV.700011.Sh_228_eegamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.700012.Sh_228_mmgamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.700013.Sh_228_ttgamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.700014.Sh_228_vvgamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.700015.Sh_228_evgamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.700016.Sh_228_mvgamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.700017.Sh_228_tvgamma_pty7_EnhMaxpTVpTy.deriv.DAOD_PHYS.e7947_s3681_r13145_p5631",
          "mc20_13TeV.345317.PowhegPythia8EvtGen_NNPDF30_AZNLO_WmH125J_Hyy_Wincl_MINLO.deriv.DAOD_PHYS.e5734_s3681_r13145_p5631",
          "mc20_13TeV.345318.PowhegPythia8EvtGen_NNPDF30_AZNLO_WpH125J_Hyy_Wincl_MINLO.deriv.DAOD_PHYS.e5734_s3681_r13145_p5631",
          "mc20_13TeV.345319.PowhegPythia8EvtGen_NNPDF30_AZNLO_ZH125J_Hyy_Zincl_MINLO.deriv.DAOD_PHYS.e5743_s3681_r13145_p5631",
          "mc20_13TeV.345061.PowhegPythia8EvtGen_NNPDF3_AZNLO_ggZH125_HgamgamZinc.deriv.DAOD_PHYS.e5762_s3681_r13145_p5631"]

# photons 
Sphotons=["mc20_13TeV.364351.Sherpa_224_NNPDF30NNLO_Diphoton_myy_50_90.deriv.DAOD_PHYS.e6452_s3681_r13145_p5631",
          "mc20_13TeV.364352.Sherpa_224_NNPDF30NNLO_Diphoton_myy_90_175.deriv.DAOD_PHYS.e6452_s3681_r13145_p5631",
          "mc20_13TeV.364353.Sherpa_224_NNPDF30NNLO_Diphoton_myy_175_2000.deriv.DAOD_PHYS.e7081_s3681_r13145_p5631",
          "mc20_13TeV.364354.Sherpa_224_NNPDF30NNLO_Diphoton_myy_2000_E_CMS.deriv.DAOD_PHYS.e7081_s3681_r13145_p5631",
          "mc20_13TeV.410389.MadGraphPythia8EvtGen_A14NNPDF23_ttgamma_nonallhadronic.deriv.DAOD_PHYS.e6155_s3681_r13145_p5631",
          "mc20_13TeV.364544.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_140_280.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364545.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_280_500.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364546.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_500_1000.deriv.DAOD_PHYS.e5938_s3681_r13145_p5631",
          "mc20_13TeV.364547.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_1000_E_CMS.deriv.DAOD_PHYS.e6068_s3681_r13145_p5631",
         ];

Sphotons=[]


def DoubleHiggs(sys=0, trig_type=0,histo_name="Mjj", massbins=None):
      global lumi, jetjet
      markerSize=1.1
      n=1
      hjet=[]
      for data in higgs2:
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
           #print(histo_name, " from ", inputroot)
           tmp=getPrediction(inputroot,histo_name)

           if (n==1):
                  hall=tmp.Clone()
           else:
                  hall.Add(tmp)

           tmp.SetTitle( getRun(data) )
           tmp.SetStats(0)
           tmp.SetLineWidth(2)
           tmp.SetLineColor( n+1 )
           tmp.SetMarkerColor( 1 )
           tmp.SetFillColor( n+1);
           hjet.append(tmp)
           n=n+1

      hall.SetMarkerSize(markerSize)
      hall.SetFillColor(0);
      hall.SetLineStyle(2)
      # smooth
      if (massbins != None):
             hall=smoothTH1( hall, massbins, 17, 3  )
      #countingErrors(hall)

      return hall, hjet




def StandardModelPredictionJZ(sys=0, trig_type=0,histo_name="Mjj", massbins=None):
      global lumi, jetjet
      markerSize=1.1
      n=1
      hjet=[]
      for data in jetjet:
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
           #print(histo_name, " from ", inputroot)
           tmp=getPrediction(inputroot,histo_name)

           if (n==1):
                  hall=tmp.Clone()
           else: 
                  hall.Add(tmp) 

           tmp.SetTitle( getRun(data) )
           tmp.SetStats(0)
           tmp.SetLineWidth(2)
           tmp.SetLineColor( n+1 )
           tmp.SetMarkerColor( 1 )
           tmp.SetFillColor( n+1);
           hjet.append(tmp)
           n=n+1

      hall.SetMarkerSize(markerSize) 
      hall.SetFillColor(0);
      hall.SetLineStyle(2)

      # smooth
      if (massbins != None): 
             hall=smoothTH1( hall, massbins, 17, 3  )
      #countingErrors(hall)


      return hall, hjet 



# get map: run vs lumiweight
def getLumiWeights():
      global lumi,jetjet,sttbar, swjets, Sdiboson,Sphotons 
      print("Get lumi weighs as map")
      lumiWeighs={}
      mcList=jetjet+sttbar+swjets+Sdiboson+Sphotons 

      for d in range(0,len(mcList)):
           run=int(getRun(mcList[d])) 
           inputroot="../analysis/out/t1/sys0/mc20/"+str(run)+"/data.root"
           rf=TFile(inputroot)
           nCPU=(rf.Get("cpucores")).GetBinContent(2)
           m_xsec=( (rf.Get("CrossSection")).GetBinContent(2) )/nCPU
           LumiWeight  = m_xsec*lumi
           rf.Close()
           lumiWeighs[run]=LumiWeight
           print("Lumi weigths for run=", run, " = ", LumiWeight)
      return lumiWeighs; 


# get SM prediction for systematic and event type
# https://danikam.github.io/2019-08-19-usatlas-recast-tutorial/09-scaling/index.html
def StandardModelPrediction(sys=0, trig_type=0,histo_name="Mjj", massbins=None):
      global lumi,jetjet,sttbar, swjets, Sdiboson,  Sphotons 
      markerSize=1.1

      # https://twiki.cern.ch/twiki/bin/view/AtlasProtected/SUSYMCSampleQCD 
      data=jetjet[1]
      inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
      print(histo_name, " from ", inputroot)
      hjet = getPrediction(inputroot,histo_name)
      # get the rest
      n=1
      for d in range(2,len(jetjet)):
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(jetjet[d])+"/data.root"
           #print(histo_name, " from ", inputroot)
           hjet.Add(getPrediction(inputroot,histo_name))
           n=n+1
           #if (n>0): break
      hjet.SetTitle("QCD Jets")
      hjet.SetStats(0)
      hjet.SetLineWidth(2)
      hjet.SetLineColor( 12 )
      hjet.SetMarkerColor( 1 )
      hjet.SetMarkerSize(markerSize)
      hjet.SetFillColor(12);

      # hjet.Print("All")


      # smooth
      if (massbins != None):  hjet=smoothTH1( hjet, massbins, 17, 3  )
      #countingErrors(hjet )

      n=0
      for data in sttbar:
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
           if (n==0): ttbar= getPrediction(inputroot,histo_name) 
           else: ttbar.Add(getPrediction(inputroot,histo_name))
           n=n+1
      ttbar.SetTitle("t#bar{t}+ single")
      ttbar.SetStats(0)
      ttbar.SetLineWidth(2)
      ttbar.SetLineColor( 9 )
      ttbar.SetMarkerColor( 1 )
      ttbar.SetMarkerSize(markerSize)
      ttbar.SetFillColor(9);
      #countingErrors(ttbar)

      n=0
      for data in swjets:
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
           if (n==0): wzjets= getPrediction(inputroot,histo_name)
           else: wzjets.Add(getPrediction(inputroot,histo_name))
           n=n+1
      wzjets.SetTitle("W/Z + jets")
      wzjets.SetStats(0)
      wzjets.SetLineWidth(2)
      wzjets.SetLineColor( 31 )
      wzjets.SetMarkerColor( 1 )
      wzjets.SetMarkerSize(markerSize)
      wzjets.SetFillColor(31);
      #countingErrors(wzjets)

      n=0
      for data in Sdiboson:
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
           if (n==0): dibosons= getPrediction(inputroot,histo_name)
           else: dibosons.Add(getPrediction(inputroot,histo_name))
           n=n+1
      dibosons.SetTitle("Di-bosons")
      dibosons.SetStats(0)
      dibosons.SetLineWidth(2)
      dibosons.SetLineColor( 6 )
      dibosons.SetMarkerColor( 1 )
      dibosons.SetMarkerSize(markerSize)
      dibosons.SetFillColor( 6 );
      #countingErrors(dibosons)

      photons=None

      """
      n=0
      for data in Sphotons: 
           inputroot="../analysis/out/t"+str(trig_type)+"/sys"+str(sys)+"/mc20/"+getRun(data)+"/data.root"
           if (n==0): photons= getPrediction(inputroot,histo_name)
           else: photons.Add(getPrediction(inputroot,histo_name))
           n=n+1
      photons.SetTitle("#gamma+jets")
      photons.SetStats(0)
      photons.SetLineWidth(2)
      photons.SetLineColor( 8 )
      photons.SetMarkerSize(markerSize)
      photons.SetMarkerColor( 1 )
      photons.SetFillColor( 8);
      #countingErrors(photons)
      """

      # add them
      hall=hjet.Clone()
      hall.SetTitle("SM prediction")
      hall.SetStats(0)
      hall.SetLineWidth(3)
      hall.SetLineColor( 2 )
      hall.SetMarkerColor( 1 )
      hall.SetFillColor( 2 );
      hall.SetMarkerSize(markerSize)

      hall.Add(ttbar)
      hall.Add(wzjets)
      hall.Add(dibosons)
      hall.Add(wzjets)
      hall.Add(dibosons)
      #hall.Add(photons)

      #countingErrors(hall)

      #sys.exit()
      return [hall, hjet, ttbar, wzjets, dibosons,photons]



# convert histogram to TGraph
def TH1toTGraphError(h1):

    g1 = TGraphErrors()
    for i in range(h1.GetNbinsX()):
        y = h1.GetBinContent(i+1)
        ey = h1.GetBinError(i+1)
        x = h1.GetBinCenter(i+1)
        ex = h1.GetBinWidth(i+1)/2.0
        g1.SetPoint(i, x, y)
        g1.SetPointError(i, ex, ey)

    g1.SetMarkerColor(1)
    g1.SetMarkerStyle(20)
    g1.SetMarkerSize(0.5)

    # g1->Print();
    return g1


def TH1Error2Zero(h1):
    for i in range(h1.GetNbinsX()):
        y = h1.GetBinContent(i+1)
        ey = h1.GetBinError(i+1)
        x = h1.GetBinCenter(i+1)
        ex = h1.GetBinWidth(i+1)
        h1.SetBinContent(i+1, y)
        h1.SetBinError(i+1, ey)
        if (y > 0):
            h1.SetBinError(i+1, 0)
        else:
            h1.SetBinError(i+1, 0)

# macro to divide by bin width taking into account Nr of cores
def getBinSize(fdata):
   xbins=fdata.Get("bins_m")
   nnn=fdata.Get("cpucores"); 
   nCPU=nnn.GetBinContent(2)
   print ("Nr of cores=",nCPU)
   xbins.Scale(1.0/nCPU)
   bins=xbins.Clone();
   bins.SetDirectory(0)
   TH1Error2Zero(bins)
   #bins.Print("all")
   return bins


import numpy as np
from scipy.signal import savgol_filter

# smooth MC histogram. Use only histograms after bin division!
# it take histogram and bin sizes
# plus difine window and polynomial for 
# @return: smoothed histogram before division
def smoothTH1(MCTOT, bins,swindow=13, spoly=3):

  # bins.Print("All")

  MCTOT_BINS=MCTOT.Clone()
  MCTOT_BINS.Divide(bins)
  #print("Before div =",MCTOT.GetBinContent(10));
  #print("Bin after div=",MCTOT_BINS.GetBinContent(10), " Bin size=",bins.GetBinContent(10));

 
  # apply smoothing ----------------------------------- 
  gr=TH1toTGraphError(MCTOT_BINS)
  x_buff = gr.GetX()
  y_buff = gr.GetY()
  N = gr.GetN()
  x_buff.SetSize(N)
  y_buff.SetSize(N)
  # Create arrays from buffers, copy to prevent data loss
  x = np.array(x_buff,copy=True)
  y = np.array(y_buff,copy=True)

  #x = np.linspace(0,2*np.pi,100)
  #y = np.sin(x) + np.random.random(100) * 0.2
  yhat = savgol_filter(y, swindow, spoly) # window size 13, polynomial order 3 

  # with bin division
  smooth_gev=MCTOT_BINS.Clone()
  smooth_gev.Reset()
  smooth_gev.SetLineColor( ROOT.kRed )

  # no bin division
  smooth=MCTOT.Clone()
  smooth.Reset()

  g1=TGraphErrors()
  for i in range(MCTOT_BINS.GetNbinsX()):
     y=MCTOT_BINS.GetBinContent(i+1)
     y=yhat[i] # smoothed  
     ey=MCTOT_BINS.GetBinError(i+1)
     x=MCTOT_BINS.GetBinCenter(i+1)
     ex=MCTOT_BINS.GetBinWidth(i+1)/2.0
     #g1.SetPoint(i,x,y)
     #g1.SetPointError(i,ex,ey)

     if (i==0): print("Smoothed =", y);

     if (y>0):
        # smoothed histogram
        smooth_gev.SetBinContent(i+1,y)
        smooth_gev.SetBinError(i+1,ey)
        # smooth but without bin division
        yevents= y*MCTOT_BINS.GetBinWidth(i+1);
        #smooth.SetBinContent(i+1, yevents)
        #smooth.SetBinError(i+1,ey*MCTOT_BINS.GetBinWidth(i+1))

        yrandom=yevents
        if (yevents>10):
            yrandom=gRandom.Gaus(yevents, sqrt(yevents));
        # randomize after smoothing
        smooth.SetBinContent(i+1, yrandom)

        #xerr=y*MCTOT_BINS.GetBinError(i+1);
        # original error
        smooth.SetBinError(i+1, 2*MCTOT.GetBinError(i+1)  )

        # increase the error to match exiting spikes that came from merging samples 
        # this is about x7 by looking at these spikes. 
        smooth.SetBinError(i+1, 20*sqrt( yrandom ))
        #if (yevents<2.99):
        #           smooth.SetBinError(i+1,yrandom )


  # print("Bin after=",smooth.GetBinContent(10));


  final=smooth.Clone()
  final_BINS=final.Clone()
  final_BINS.Divide(bins)

  #print("Bin after=",final_BINS.GetBinContent(10));
  return final

# calculate distance between data and all BSM histograms (as list)
def distanceHistograms(data, bsm=[]):
    Xdata=data.Clone()
    Xdata.Scale(1.0/Xdata.Integral())
    dMean=Xdata.GetMean()
    dData=dMean+3*Xdata.GetRMS()
    kk=0
    ks=0
    rms=0
    for b in bsm:
       #ks=ks+data.KolmogorovTest(b)
       #ks=ks+data.KolmogorovTest(b, "N") # with normalisation 
       Xbs=b.Clone()
       print("Entries=",Xbs.GetEntries(), " mean=",Xbs.GetMean())
       if (Xbs.GetEntries()<100): continue # no low statistics  

       if (Xbs.Integral()>0): Xbs.Scale(1.0/Xbs.Integral())
       #dmean=Xbs.GetMean() - dMean;
       ks=ks+Xbs.GetMean()
       rms=rms+Xbs.GetRMS()
       kk=kk+1 
       #countingErrorsCorrect( b )
       #ks=ks+Xdata.Chi2Test(Xbs,"CHI2/NDF") # "NORM" 
       #ks=ks+data.Chi2Test(b,"CHI2") # returns CHI value 
    if (kk>0):
      ks=ks/float(kk)
      rms=rms/float(kk)
    print("Average=",ks,"  sigma=",rms)
    return [( ks - rms ), dData] # this is BSM 


# calculate distance between data and SM histogram
def distanceHistogramsSM(data, sm):
    Xdata=data.Clone()
    dMean=Xdata.GetMean()
    dData=dMean+3*Xdata.GetRMS()

    Mdata=sm.Clone()
    mMean=Mdata.GetMean()
    mData=mMean+3*Mdata.GetRMS()
   
    return [dData,  mData] 

def GetZVal (p, excess) :
  #the function normal_quantile converts a p-value into a significance,
  #i.e. the number of standard deviations corresponding to the right-tail of 
  #a Gaussian
  if excess :
    zval = ROOT.Math.normal_quantile(1-p,1);
  else :
    zval = ROOT.Math.normal_quantile(p,1);

  return zval


# calculate significance
def signif(S,B):
     # print("Calculate significance for B=",B," and S=",S)
     if (S<10e-6): return S/math.sqrt(B) # avoid domain problem 
     else:  return math.sqrt(2* ( (S+B)*math.log(1+ float(S)/B)-S ) )


