#import sys
#if sys.version_info >= (3, 0):
#    sys.stdout.write("Sorry, requires Python 2.x, not Python 3.x\n")
#    sys.exit(1)


# Max events to look at
MaxEvents=100000 

from ROOT import gROOT,gPad,gStyle,TCanvas,TSpline3,TFile,TLine,TLatex,TAxis,TLegend,TPostScript
from ROOT import TH2D,TF2,TArrow,TCut,TPad,TH1D,TF1,TObject,TPaveText,TGraph,TGraphErrors,TGraphAsymmErrors
from ROOT import TGraph2D,TTree,TMultiGraph,TBranch,gSystem,gDirectory
from ROOT import TPaveStats,TProfile2D 
import sys
sys.path.append("modules/")
#from AtlasStyle import *
#from AtlasUtils import *
import math
import ROOT
import os,sys
#sys.path.append("nnet/")
#import bpnn

print ('Number of arguments:', len(sys.argv), 'arguments.') 
print ('Argument List:', str(sys.argv))
print ('Use as: script.py channel ()') 

channel=1 
myinput="interactive"
if (len(sys.argv) ==2):
    channel = sys.argv[1] 
if (len(sys.argv) == 3):
   channel = sys.argv[1]
   myinput = sys.argv[2]

print ("Mode=",myinput)
print("Validata RMM matrix for channel = ",channel)

out=[0]
gROOT.Reset()
figdir="figs/"
fname=os.path.basename(__file__)  
epsfig=figdir+fname.replace(".py",".eps")
epsfig=epsfig.replace("data","data_"+str(channel))

name="projection"
#name="profile"
nameX=""
nameY=""
Ymin=0.0
Ymax=500000
Xmin=0
Xmax=6.0 
ZSmin=0.00000001
ZSmax=0.5

######################################################
gROOT.SetStyle("Plain");
gStyle.SetLabelSize(0.035,"xyz");
c1=TCanvas("c_massjj","BPRE",10,10,800,700);
c1.Divide(1,1,0.008,0.007);
ps1 = TPostScript( epsfig,113)
c1.SetGrid();

c1.cd(1);
gPad.SetLogy(0)
gPad.SetLogz(1)
gPad.SetTopMargin(0.05)
gPad.SetBottomMargin(0.1)
gPad.SetLeftMargin(0.1)
gPad.SetRightMargin(0.15)
gPad.SetGrid();
gPad.SetTicks(2,0)

# data
proc=["../out/tev13.6pp_pythia8_ttbar_2lep.root"]

rfile=[]
print ("Look at ", len(proc) ," processes using RMM", proc) 

for i in proc:
     rfile.append(ROOT.TFile.Open(i))
     print(i) 

rfile[0].ls()


dimensions=(rfile[0]).Get("dimensions");
#h_cpucores=(rfile[0]).Get("cpucores");
print("Nr of used CPU=")
cpucores=1; # int(h_cpucores.GetBinContent(2))
print("Nr of CPU used=",cpucores)

dimensions.Print("All")

maxNumber=int(dimensions.GetBinContent(2)/cpucores)
maxTypes=int(dimensions.GetBinContent(3)/cpucores)
mSize=int(dimensions.GetBinContent(4)/cpucores)
print ("maxNumber=",maxNumber," maxTypes=",maxTypes," mSize=",mSize) 


mSize=maxTypes*maxNumber+1;
hhD = TProfile2D("profile", "profile", mSize, 0, mSize, mSize, 0, mSize, 0, 1000);
names=["MET","j", "b", "#mu", "e", "#gamma"]


Names1=[]
Names1.append(names[0]);
for h in range(1,maxTypes+1,1):
       for i in range(1,maxNumber+1):
                 Names1.append(names[h]+"_{"+str(i)+"}");
Names2=[]
for i in range(len(Names1)):
         Names2.append(Names1[i]);
Names1= Names1[::-1]

print(Names1) 

for h in range(mSize):
      for w in range(mSize):
        i1=h;
        i2=w;
        # hhD.Fill(Names2[i1],  Names1[i2], 0.0);
        hhD.Fill(i1,  i2, 0.0);

hhD.SetTitle("")
hhD.SetStats(0)
hhD.GetZaxis().SetRangeUser(ZSmin,ZSmax);
hhD.GetZaxis().SetLabelOffset(0.02)
hhD.GetYaxis().SetLabelOffset(0.02)
hhD.GetXaxis().SetLabelOffset(0.02)
gStyle.SetPaintTextFormat(".0e");

kk=0;
events=0;
inputs=0
outputs=0
inpp=[]
out=[]

for event in rfile[0].inputNN:
       NN=(event.proj).size()
       a=event.proj
       inx1=event.proj_index1
       inx2=event.proj_index2
       Trun = 1 # event.run
       Tevent=kk # event.event
       Tweight=1.0; # event.weight # for MC with weigths
       weight=Tweight;

       for i3 in range(NN):
              w=inx1[i3];
              h=inx2[i3];
              i1=w;
              i2=mSize-h-1;
              val=float(a[i3])
              #print Names2[h],  Names1[w],float(val)
              #hhD.Fill(Names2[i1],  Names1[i2],val);
              hhD.Fill(i1, i2, float(val));

       kk=kk+1
       if (kk%500==0): print ("Process=",kk) 

       if (kk>MaxEvents):
                    print ("Stop after ",MaxEvents); break;

hhD.Draw("colz")
#hhD.Draw("text0 same")

x1=0
x2=mSize 
YRMIN=mSize
YRMAX=0

ar7=TLine(x1,YRMIN,x2,YRMAX);
ar7.SetLineWidth(2)
ar7.SetLineStyle(2)
ar7.SetLineColor(0)
ar7.Draw("same")

#myText(0.75,0.13,10,0.09,"T"+str(channel))


print (epsfig) 
gPad.RedrawAxis()
c1.Update()
ps1.Close()
if (myinput != "-b"):
              if (input("Press any key to exit") != "-9999"):
                         c1.Close(); sys.exit(1);




