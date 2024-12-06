# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import numpy
import sys,os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
import matplotlib.pyplot as plt
####### Deep learning libraries
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_auc_score, auc,
                             precision_score, recall_score, roc_curve, precision_recall_curve,
                             precision_recall_fscore_support, f1_score,
                             precision_recall_fscore_support)

RANDOM_SEED = 101
# fix random seed for reproducibility
numpy.random.seed(RANDOM_SEED)

# read input CSV files to DataFrame
def ReadData(inputData):
      print("Read=",inputData)
      df = pd.read_csv(inputData, compression='gzip', header=0, sep=',')
      #print("DF size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)
      #print("Skip run, event and weight columns..")
      #df.drop(['Run', 'Event', 'Id'], axis = 1)
      del df['Run']
      del df['Event']
      del df['Weight']
      #print("DF size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)
      file0="../train/columns_with_0_10j10b5rest.txt"
      #print("Read common 0 columns from ",file0)
      dcol0=pd.read_csv(file0,header = None)
      #print ("-> Experimental: Drop columns with 0")
      col0=dcol0[dcol0.columns[0]]
      df=df.drop(col0, axis = 1)
      df=df.drop('Label', axis = 1)

      # get mass from file name
      mass_value=-1
      start = inputData.find("X") + 1
      end = inputData.find("GeV")
      if (start>0 and end>0):
                   mass_value = float(inputData[start:end])
                   print("Found mass=",mass_value);

      CM_ENERGY=136000.0

      # add mass / CM for signal, but fill random for data 
      print("Nr of columns=",df.shape[1]);
      mass=np.empty(df.shape[0]); 
      if (mass_value>0): mass.fill(mass_value/CM_ENERGY) 
      else:  mass=np.random.uniform(low=0.0, high=2000/CM_ENERGY, size=(df.shape[0],))

      df['Mass'] = mass 
      #df.insert(loc=1, column='Mass', value=mass)
      print("Nr of columns after adding mass=",df.shape[1]);


      #print("Total zero-cells removed=",len(dcol0))
      print("Input=",inputData," DF size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)
      return df

# read BSM (MG5) first
df1=ReadData("out/pythia8_X500GeV_HH2bbll_data100percent.csv.gz")
df2=ReadData("out/pythia8_X700GeV_HH2bbll_data100percent.csv.gz")
df3=ReadData("out/pythia8_X1000GeV_HH2bbll_data100percent.csv.gz")
df4=ReadData("out/pythia8_X1500GeV_HH2bbll_data100percent.csv.gz")
df5=ReadData("out/pythia8_X2000GeV_HH2bbll_data100percent.csv.gz")



# combine all 
df=pd.concat([df1,df2,df3,df4,df5])
print("Final after append: size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)
# sys.exit()


# read SM
dfSM=ReadData("out/tev13.6pp_pythia8_ttbar_2lep_data10percent.csv.gz")

# trim to signal sample
dfSM= dfSM.tail( len(df)  )
print("After trim SM to signal size=", "DF size=",dfSM.size," DF shape=",dfSM.shape," DF dimension=",dfSM.ndim)
#sys.exit()


print("Train on 10% of data, as for the AD filter")
SplitSize=0.9 
print("## Data Preprocessing:")
print("-> Validation fraction=",SplitSize," Training fraction=",1-SplitSize)
X_train, X_valid = train_test_split(df, test_size=SplitSize, random_state = RANDOM_SEED, shuffle=True)

print('Training data size   :', X_train.shape)
print('Validation data size :', X_valid.shape)

# creating 1-d array
Y_train = np.ones(X_train.shape[0])
Y_valid = np.ones(X_valid.shape[0])
print("Size of outputs=",len(Y_train) )


### deal with SM
X_SM_train, X_SM_valid = train_test_split(dfSM, test_size=SplitSize, random_state = RANDOM_SEED, shuffle=True)
# ouput must be 0
Y_SM_train = np.zeros(X_SM_train.shape[0])
Y_SM_valid = np.zeros(X_SM_valid.shape[0])

# Now combine SM with BSM
print('BSM Training data size   :', X_train.shape)
print('BSM Validation data size :', X_valid.shape)
print('SM Training data size   :', X_SM_train.shape)
print('SM Validation data size :', X_SM_valid.shape)

X_train=np.append(X_train, X_SM_train,  axis=0)
Y_train=np.append(Y_train, Y_SM_train,  axis=0)

X_valid=np.append(X_valid, X_SM_valid, axis=0)
Y_valid=np.append(Y_valid, Y_SM_valid, axis=0)

print('Training data size after append  :', X_train.shape)
print('Validation data size after append :', X_valid.shape)


#print(type(X_train))

# sys.exit()


# Modeling
# No of Neurons in each Layer 
input_dim = X_train.shape[1] 
print("input_dim :", X_train.shape[1])  # Check the number of features (input dimension)

MaxEpochs=200
print("Max epochs for =", MaxEpochs)

encoding_dim =  800
encoding_dim2 =  400
# create model
model = Sequential()
model.add(Dense(encoding_dim, input_dim=input_dim, activation='relu'))
model.add(Dense(encoding_dim2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# patience 5
callback = EarlyStopping(monitor='loss', patience=5) 
# Fit the model
model.fit(X_train, Y_train, epochs=MaxEpochs, batch_size=10, shuffle=True, callbacks=[callback], verbose=2)
# evaluate the model
scores = model.evaluate(X_train, Y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# serialize model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk: models/model.h5")
 
# later...
 
# load json and create model
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_valid, Y_valid, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
