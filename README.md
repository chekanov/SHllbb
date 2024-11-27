Full analysis chain for analysis of HepSim files using Autoenecoder and Backpropogation with RMM.

## Autoencoder example

Use the directory: ana_truth
Setup C++/ROOT as : source msetup.sh


Read ProMC files. The list of files is given in file data.in

source msetup.sh
 make


Run over all files using A_RUN which executes ana.cc.
The output ROOT files goes to "out" directory

```
   ./A_RUN_ttbar - ttbar background
   ./A_RUN_wzjets - Z+jets
   ./A_RUN_X2hh - X-> hh (MG5, all decays)
   ./A_RUN_X2HH - X-> HH (Pythia8, ZZ+bb decays, Z to ll)
   ./A_RUN_X2SH - X-> SH (Pythia8, ZZ+bb decays, Z to ll)  S has the mass m(X)/2
```

The data are:

```
SM
https://atlaswww.hep.anl.gov/hepsim/info.php?item=382
https://atlaswww.hep.anl.gov/hepsim/info.php?item=384
```

```
BSM:
https://atlaswww.hep.anl.gov/hepsim/info.php?item=383
https://atlaswww.hep.anl.gov/hepsim/info.php?item=385
https://atlaswww.hep.anl.gov/hepsim/info.php?item=386
```


You need to download this data so you can run these scripts.
  

2) Extract 10% of data and save as CSV file. Do this:

```
    cd ./prepare
   ./A_RUN_PREPARE_CSV
```

   The *csv.gz files will be in "out" directory

3) Do training using this csv.gz file. Go to:

```
   cd train/
   ./A_RUN
```

   This script runs "arun_autoencoder.py" which reads cvs.gz files
   and train autoencoder.

   The training model will show up in train/models

4) Final run to process all data using the trained model. In the main directory run

```
   ./A_RUN_ANALYSIS - for ttbar
   ./A_RUN_ANALYSIS_X2HH - for HH (Pythia8, ZZ+bb decays, one Z to ll)
   ./A_RUN_ANALYSIS_X2SH - for SH (Pythia8, ZZ+bb decays, S goes to ZZ, one Z to ll)
```

  The files with losses and Mjj will be in "out/*ADFilter* directory


## Backpropogation example 

Go to "train_bkprop/" directory and train NN using "A_RUN". The model will me saved to "model/".

Then  process data with trained neural net using the scripts "*ANALYSIS* 

It fills the histogram "Score" which is 0 for ttbar and 1 for BSM. The outputs goes to "out/*BACKprop.root" histograms. The inavrainat mass for BSM is selected when score>0.5


## Making plots

X->HH:
```
   loss_cut_Hh.py  - loss distribution for AD
   backprop_score_Hh.py - backpropogation score
   kin_mbb_Hh.py  - X->SH before anything
   kin_mbb_Hh_after_bkprop.py - after backpropogation
   kin_mbb_Hh_after_ad.py - after AD filter
```

   X->SH:
``` 
   loss_cut_Sh.py  - loss distribution for AD
   backprop_score_Sh.py - backpropogation score
   kin_mbb_Sh.py  - X->SH before anything
   kin_mbb_Sh_after_bkprop.py - after backpropogation
   kin_mbb_Sh_after_ad.py - after AD filter
```   

-- S.Chekanov (ANL)

 
