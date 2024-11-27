# Anomaly detection https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.3542&rep=rep1&type=pdf
# Task: Run autoencoder over imput data and store model at the end.


import os,sys
sys.path.append("modules/")
from global_module import *
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)


print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
n = len(sys.argv)
if (n != 2):
      print ("No arguments!. Need at least 1 input file in csv.zip")
      sys.exit()
inputData=sys.argv[1]

# Model name to save
from pathlib import Path   
tail = Path(inputData).name 
modelName="./models/"+tail.replace(".csv.gz","")
figsDir ="./figs/"+tail.replace(".csv.gz","")
print("Train model = ",modelName) 
print("Figures in = ", figsDir )

if not os.path.exists(figsDir):
    os.makedirs(figsDir)


# Nr of epochs
nb_epoch = 1000  
print("Nr of epochs=",nb_epoch)

batch_size = 100 
print("Bunch size=",batch_size)

# keep summary of training 
summF=open(modelName+".txt","w")
summF.write("### Summary of trainning\n")


##########  AutoEncoders
# Data Preprocessing
import pandas
import matplotlib
import seaborn
import tensorflow
import tensorflow as tf
##########  AutoEncoders
import pickle
print('Numpy version      :' , numpy.__version__)
print('Pandas version     :' ,pandas.__version__)
print('Matplotlib version :' ,matplotlib.__version__)
print('Seaborn version    :' , seaborn.__version__)
print('Tensorflow version :' , tensorflow.__version__)
# print('Keras version      :' , keras.__version__)

import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
import matplotlib.pyplot as plt
plt.rcdefaults()
from pylab import rcParams
import seaborn as sns
import datetime
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplo. Fix Invalid DISPLAY variable 
from matplotlib import pyplot as plt
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



# random seeds fixed
RANDOM_SEED = 101
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED*2)
random.seed(RANDOM_SEED*3)
print("Use fixed seed=",RANDOM_SEED)
os.environ["OMP_NUM_THREADS"] = "1"
physical_devices = tf.config.list_physical_devices('CPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


print("Reading=",inputData)
df = pd.read_csv(inputData, compression='gzip', header=0, sep=',')
print("DF size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)
#df = df.astype(np.float16)
#print(" Type=",df.dtypes)
#print("Convert to range= Min",numpy.finfo(numpy.float16).min," Max=",numpy.finfo(numpy.float16).max)


print("Skip run, event and weight columns..") 
#df.drop(['Run', 'Event', 'Id'], axis = 1)
del df['Run']
del df['Event']
del df['Weight']
print("DF size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)


# do you want to drop columns based on common  data?
IsReadCommonEmptyColumns=1

# 1 drop colums based on common vector
# 2 drop colums as found by the current dataframe
# 0 do not drop anything

if (IsReadCommonEmptyColumns==1):
   # file0="columns_with_0.txt"
   file0="columns_with_0_10j10b5rest.txt"
   print("Read common 0 columns from ",file0)
   dcol0=pd.read_csv(file0,header = None)
   print ("-> Experimental: Drop columns with 0")
   col0=dcol0[dcol0.columns[0]]
   df=df.drop(col0, axis = 1)
   print("Total zero-cells removed=",len(dcol0))
elif (IsReadCommonEmptyColumns==2):
   print ("Experimental: find all columns with 0")
   col0 = df.columns[(df == 0).all()]
   print("COL=0 size=",col0.size," DF shape=",col0.shape," DF dimension=",col0.ndim)
   print(col0)
   #print ("Experimental: Drop columns with 0")
   #df=df.drop(col0, axis = 1)
   file0=modelName+"/columns_with_0.txt"
   print ("Experimental: Save columns with 0 in ",file0)
   print(type(col0))
   pd.Series(col0,index=col0).to_csv(file0, header=False, index=False)
   print ("Experimental: Restore columns with 0 from ",file0)
   dcol0=pd.read_csv(file0,header = None)
   col0=dcol0[dcol0.columns[0]]
   print ("-> Experimental: Drop columns with 0")
   df=df.drop(col0, axis = 1)
else:
   pass


print("Apply scalers and remove 0 columns: DF size=",df.size," DF shape=",df.shape," DF dimension=",df.ndim)

#print(" -> Shuffle the DataFrame rows")
#df = df.sample(frac = 1)

#xhead=df.head()
#print(xhead)

print("")
SplitSize=0.3
print("## Data Preprocessing:") 
print("-> Validation fraction=",SplitSize," Training fraction=",1-SplitSize)
X_train, X_valid = train_test_split(df, test_size=SplitSize, random_state = RANDOM_SEED, shuffle=True)
#print('X_train =', X_train.head(5))
#print('X_train type =', type(X_train['V_1'][0]))
#print('X_valid =', X_valid.head(5))
#print('X_valid type =', type(X_valid['V_1'][0]))


# If you want to remove rows with some label (0) 
# X_train = X_train[X_train['Label'] == 0]
X_train = X_train.drop(['Label'], axis=1)
y_test  = X_valid['Label']
X_valid  = X_valid.drop(['Label'], axis=1)
X_train = X_train.values
X_valid  = X_valid.values
print('Training data size   :', X_train.shape)
print('Validation data size :', X_valid.shape)
summF.write('Training data size   :'+str( len(X_train.shape) )+"\n")
summF.write('Validation data size :'+str( len(X_valid.shape) )+"\n")



# apply Standardization and MinMax?
IsStandard=False


if (IsStandard):
  print("")
  print("Data Standardization.. so that the mean of observed values is 0 and the standard deviation is 1.");
  scaler = StandardScaler() # create scaler 
  scaler.fit(X_train) # fit scaler on data 
  scaler_filename = modelName+"/StandardScaler.pkl"
  print("Save fitted StandardScaler =",scaler_filename)
  pickle.dump(scaler, open(scaler_filename, 'wb'))
  # apply transform
  X_train = scaler.transform(X_train)
  X_valid = scaler.transform(X_valid)

  print ("Data scaling.. Can be skipped since RMM [0-1]. But you ran standartisation before!")
  scaler = MinMaxScaler(feature_range=(0,1))
  scaler.fit(X_train)
  scaler_filename = modelName+"/MinMaxScaler.pkl"
  print("Save fitted MinMaxScaler =",scaler_filename)
  pickle.dump(scaler, open(scaler_filename, 'wb'))
  X_train = scaler.transform(X_train)
  X_valid = scaler.transform(X_valid)
  print("")
else:
  print("No data Standardization since RMM already have (0,1) range");
  pass

# Verify minimum value of all features
# print(X_train.min(axis=0))


# Modeling
# No of Neurons in each Layer [2602,12,6,4,6,12,2602]
input_dim = X_train.shape[1]

"""
encoding_dim = 12
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
encoder = Dense(int(4), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()
"""

encoding_dim =  800
encoding_dim2 =  400
latent_dim=200
learning_rate = 0.5e-7
#input Layer
input_layer = Input(shape=(input_dim, ))
encoder = tf.keras.layers.Dense(encoding_dim, activation=tf.nn.leaky_relu,activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
# encoder=tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(encoding_dim2, activation=tf.nn.leaky_relu)(encoder)
encoder = tf.keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(encoder)
# Decoder
decoder = tf.keras.layers.Dense(encoding_dim2, activation=tf.nn.leaky_relu)(encoder)
# decoder=tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation=tf.nn.leaky_relu)(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation=tf.nn.leaky_relu)(decoder)
#Autoencoder
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()



autoencoder.compile(optimizer='adam', loss='mse' )
# print(autoencoder.layers[3].get_weights())

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    mode='min',
    verbose=1,
    restore_best_weights=True 
)

# early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, baseline=3.25e-05)


# Create a callback that saves the model's weights every 10 epochs
savemode = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelName, 
    verbose=1, 
    save_freq='epoch',
    period=10);


t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        validation_data=(X_valid, X_valid), 
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=2,
                        callbacks=[early_stopping, savemode] 
                        )

t_fin = datetime.datetime.now()
print('## Finished! Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))

# Let's plot training and validation loss to see how the training went.
lines_loss=plt.plot(history.history["loss"], label="Training Loss")
lines_val_loss=plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend(loc='upper right')
plt.title('Model loss')
xlab,ylab ='Epoch', 'Loss'
plt.ylabel(xlab)
plt.xlabel(ylab)
axes = plt.gca()
axes.set_xlim([0, nb_epoch])
axes.set_ylim([0.00002,0.001])
plt.yscale('log')
plt.savefig(figsDir+'/model_loss.pdf', bbox_inches='tight')
plt.show()
SavePlotXY(figsDir+'/model_loss.csv', lines_loss, xlab,ylab)
SavePlotXY(figsDir+'/model_val_loss.csv',lines_val_loss, xlab,ylab)
plt.clf();
plt.close();


df_history = pd.DataFrame(history.history)

########### Predictions & Computing Reconstruction Error ##################
# Detect Anomalies on test data
# Anomalies are data points where the reconstruction loss is higher
# To calculate the reconstruction loss on test data, predict the test data and calculate the mean square error between the test data and the reconstructed test data.
predictions = autoencoder.predict(X_valid)

mse = np.mean(np.power(X_valid - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'Label': y_test}, index=y_test.index)
print( df_error.describe() )


# Plotting the test data points and their respective reconstruction error sets a threshold value to visualize if the threshold value needs to be adjusted.

threshold_fixed = 0.005   

groups = df_error.groupby('Label')
fig, ax = plt.subplots()
for name, group in groups:
    #print("Name=",name," group=",group)
    summF.write("Name="+str(name)+" group="+str(group)+"\n")
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Anomaly" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for normal and anomalous data")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.savefig(figsDir+'/reconstruction_error.pdf', bbox_inches='tight')
plt.show();
plt.clf();
plt.close();

print("Show train loss as histogram")
train_loss = tf.keras.losses.mae(predictions, X_valid).numpy()
# plt.hist(train_loss[None,:], bins=50)
# plt.hist(train_loss, bins=[i*0.0001 for i in range(0,400,2)], histtype='step', label='Background', color='k')
plt.hist(train_loss,  bins=200, range=(0, 0.02), histtype='step', label='Background', color='k')
#w = 100 # 100 bins 
#n = math.ceil((train_loss.max()*1.2 - train_loss.min())/w)
#plt.hist(train_loss, bins = n, histtype='step', label='Background', color='k')
plt.xlabel("Train loss")
plt.ylabel("Events")
plt.yscale('log')
SaveNumpyData(figsDir+'/train_loss_histo_data.csv', train_loss)
SavePlotHisto(figsDir+'/train_loss_histo.csv',plt.gca())
plt.savefig(figsDir+'/train_loss_histo.pdf', bbox_inches='tight')
plt.show()
plt.clf();
plt.close();


threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold calculated: ", threshold)
summF.write("Threshold calculated: "+str(threshold)+"\n")
summF.close()

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))


# https://www.tensorflow.org/tutorials/generative/autoencoder
#preds = predict(autoencoder, X_valid_scaled, threshold)
#print_stats(preds, test_labels)

# See: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#print ("serialize model to JSON")
#model_json = autoencoder.to_json()
#fm1="figs/model.json"
#with open(fm1, "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#fm2="figs/model.h5"
#autoencoder.save_weights(fm2)
#print("Saved model :",fm1,fm2)
# Write the model definition

print("Write model to: ",modelName)
autoencoder.save(modelName, save_format='tf')
print("Write figures to: ", figsDir )
print("-> Done!")


"""
# Evaluating the performance of the anomaly detection
LABELS = ["Normal","Anomaly"]
pred_y = [1 if e > threshold_fixed else 0 for e in df_error.reconstruction_error.values]
df_error['pred'] =pred_y
conf_matrix = confusion_matrix(df_error.Label, pred_y)
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig(f'figs/Confusionmatrix.pdf', bbox_inches='tight')
plt.show()
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(df_error['Label'], df_error['pred']))
print(" Recall: ",recall_score(df_error['Label'], df_error['pred']))
print(" Precision: ",precision_score(df_error['Label'], df_error['pred']))




# change X_tes_scaled to pandas dataframe
data_n = pd.DataFrame(X_valid_scaled, index= y_test.index, columns=numerical_cols)
def compute_error_per_dim(point):
    initial_pt = np.array(data_n.loc[point,:]).reshape(1,9)
    reconstrcuted_pt = autoencoder.predict(initial_pt)
    return abs(np.array(initial_pt  - reconstrcuted_pt)[0])

outliers = df_error.index[df_error.reconstruction_error > threshold_fixed].tolist()
print(outliers)

RE_per_dim = {}
for ind in outliers:
    RE_per_dim[ind] = compute_error_per_dim(ind)
RE_per_dim = pd.DataFrame(RE_per_dim, index= numerical_cols).T
print(RE_per_dim.head())

"""

sys.exit(1);


