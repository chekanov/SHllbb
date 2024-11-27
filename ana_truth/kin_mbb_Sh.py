import sys
sys.path.append("modules/")
import math

# what to plot?
name="Mbb" 
#name="jet_pt" 
nameX="M_{bb} [GeV]"
print(name, nameX)


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
print('Use as: script.py -b 0 (or 1,2)')
myinput="interactive"
xdir=""
cross=0
expected_pb=1000.
# trigger type
myinput="interactive"
trig_type="1"
if (len(sys.argv) ==2):
   trig_type=sys.argv[1]
if (len(sys.argv) ==3):
   trig_type=sys.argv[1]
   myinput = sys.argv[2]
if (len(sys.argv) == 4):
   trig_type=sys.argv[2]
   myinput = sys.argv[3]
   myinput = sys.argv[4]


# import atlas styles
from AtlasStyle import *
from AtlasUtils import *
from initialize  import *
from global_module import *



gROOT.Reset()
figdir="figs/"
fname=os.path.basename(__file__)
# fname=fname.replace(".py","_"+trig_type+".py");
epsfig=figdir+(fname).replace(".py",".eps")

# plot ranges
nameY="Events"
Ymin=0.1
Ymax=500*1000 -10 
Xmin=0
Xmax=1000.-1

# sideband normalisation
XminSide=0
XmaxSide=5000


NN=0

######################################################
gROOT.SetStyle("Plain");

gStyle.SetLabelSize(0.035,"xyz");
c1=TCanvas("c_massjj","BPRE",10,10,500,500);
c1.Divide(1,1,0.008,0.007);
ps1 = TPostScript( epsfig,113)

c1.cd(1);
gPad.SetLogy(0)
gPad.SetLogx(0)
gPad.SetTopMargin(0.05)
gPad.SetBottomMargin(0.12)
gPad.SetLeftMargin(0.14)
gPad.SetRightMargin(0.04)


#name="mass_bb_zmass"
name="mass_bb"
ffD=TFile("out/tev13.6pp_pythia8_ttbar_2lep.root")
hhD=ffD.Get(name)
cross=ffD.Get("cross");
xsec=cross.GetBinContent(1)
lumi=float(cross.GetBinContent(4))
print("Cross=",xsec," lumi=",lumi)

Imin=hhD.FindBin(XminSide)
Imax=hhD.FindBin(XmaxSide)
xsum=hhD.Integral(Imin,Imax)
#ffD.ls()

CurrentLumuFB=lumi/1000.0
Scale=ExpectedLumiFB/CurrentLumuFB;
hhD.Scale(Scale)
lumi=ExpectedLumiFB*1000;


hhD.SetAxisRange(Ymin, Ymax,"y");
hhD.SetAxisRange(Xmin, Xmax,"x");
hhD.SetTitle("")
hhD.SetMarkerColor(1)
hhD.SetMarkerSize(0.5)
hhD.SetMarkerStyle(20)
hhD.SetStats(0)
hhD.SetFillStyle(3001); 
hhD.SetFillColor(4); 
hhD.Draw("same histo")



ffZ=TFile("out/tev13.6pp_pythia8_wzjet_2lep.root")
hhZ=ffZ.Get(name)
crossZ=ffZ.Get("cross");
xsecZ=crossZ.GetBinContent(1)
lumiZ=float(crossZ.GetBinContent(4))
print("Cross=",xsecZ," lumi=",lumiZ)
CurrentLumuZ=lumiZ/1000.0
Scale=ExpectedLumiFB/CurrentLumuZ;
hhZ.Scale(Scale)
hhZ.SetFillStyle(3001);
hhZ.SetFillColor(3);
hhZ.Draw("same histo")


hhDD=hhD.Clone()
hhDD.Add(hhZ)
hhDD.SetMarkerColor(1)
hhDD.SetMarkerSize(0.8)
hhDD.SetMarkerStyle(21)
hhDD.Draw("same pe")


# Sh cannel
def getBSM_X2hh(mass):
      ffZ=TFile("out/pythia8_X"+str(mass)+"GeV_SH2bbll.root")
      hh=ffZ.Get(name)
      crossZ=ffZ.Get("cross");
      #crossZ.Print("All")  
      xsecZ=crossZ.GetBinContent(1)
      lumiZ=float(crossZ.GetBinContent(4))
      print("Cross=",xsecZ," lumi=",lumiZ," name=",name)
      CurrentLumuZ=lumiZ/1000.0
      Scale=ExpectedLumiFB/CurrentLumuZ;
      hh.Scale(Scale)
      hh.SetDirectory(0)
      hh.SetTitle(str(mass))
      hh.SetName(str(mass))
      hh_orig=hh.Clone()
      hh_orig.SetDirectory(0)
      colo=31
      scale=30000000000  
      # define colors
      if mass in xmapcolor:
                  colo=xmapcolor[mass]
      if (mass==700): scale=scale*50 
      if (mass==1000): scale=scale*500 
      if (mass==1500): scale=scale*5000 
      if (mass==2000): scale=scale*50000 
      hh.Scale(scale) # to see better  
      hh.SetTitle(str(mass))
      hh.SetName(str(mass))
      hh.SetLineColor(colo)
      hh.SetFillColor(colo);
      return [hh_orig,hh] # return orginal and shown histograms 

signals=[]
mass500_org,mass500=getBSM_X2hh(500)
mass500.Draw("same histo")
signals.append(mass500_org)

mass700_org,mass700=getBSM_X2hh(700)
mass700.Draw("same histo")
signals.append(mass700_org)

mass1000_org,mass1000=getBSM_X2hh(1000)
mass1000.Draw("same histo")
signals.append(mass1000_org)

mass1500_org, mass1500=getBSM_X2hh(1500)
mass1500.Draw("same histo")
signals.append(mass1500_org)

mass2000_org, mass2000=getBSM_X2hh(2000)
mass2000.Draw("same histo")
signals.append(mass2000_org)


# mass500_org.Print("All")


MaxBin=120 
Idx=hhD.FindBin(MaxBin)
print("Bin with maximum entries=",Idx)
for i in range(len(signals)):
         his=signals[i]
         Idx=his.FindBin(MaxBin)
         N_signal = his.GetBinContent( Idx );
         N_bkg=hhD.GetBinContent( Idx );
         Sign=signif(N_signal, N_bkg)
         print("Mass=",his.GetTitle(), " Sign=",Sign)


leg2=TLegend(0.4, 0.55, 0.95, 0.8);
leg2.SetBorderSize(0);
leg2.SetTextFont(62);
leg2.SetFillColor(10);
leg2.SetTextSize(0.035);
leg2.AddEntry(hhDD,"SM background","lp")
leg2.AddEntry(hhD,mcTT,"lfp")
leg2.AddEntry(hhZ,mcWZ,"lfp")
leg2.AddEntry(mass500,"X#rightarrow SH, M_{X}=500-2000 scaled","lfp")
leg2.AddEntry(mass500,"M(S)=M(X)/2","")

#for k in mcprediction:
#         leg2.AddEntry(k,k.GetTitle(),"l")
leg2.Draw("same")

#hhD.SetAxisRange(Ymin, Ymax,"y");
#hhD.SetAxisRange(Xmin, Xmax,"x");

Lumi=" %.0f fb^{-1}" % (lumi/1000.)
intLUMI="#int L dt = "+Lumi
myText(0.6,0.85,1,0.035,intLUMI) 
#myText(0.3,0.89,1,0.03,"pp #sqrt{s}=13 TeV")
#myText(0.7,0.61,2,0.04,"all jets")

#myText(0.19,0.82,1,0.04, UsedData0Run23)

ax=hhD.GetXaxis(); ax.SetTitleOffset(0.8)
ax.SetTitle( nameX );
ax.SetTitleSize( 0.05 );
ay=hhD.GetYaxis(); ay.SetTitleOffset(0.8)
ay.SetTitle( nameY );
ax.SetTitleOffset(1.1); ay.SetTitleOffset(1.2)
ay.SetTitleSize( 0.05 );
ax.Draw("same")
ay.Draw("same")


# ATLASLabel(0.2,0.9,0.17,0.02)

#myText(0.7,0.9,2,0.06, getTriggerLabel(trig_type) )



print(epsfig)
gPad.RedrawAxis()
c1.Update()
ps1.Close()
if (myinput != "-b"):
              if (input("Press any key to exit") != "-9999"): 
                         c1.Close(); sys.exit(1);


