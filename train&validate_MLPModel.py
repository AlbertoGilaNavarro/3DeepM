import pandas as pd
import numpy as np

import myNeuralNetworks

#%% Load the data
#You must create your compressed 2D-array starting with the 3D-arrays provided in the dataset. Check the script arraysCompression.py
trainX=pd.read_table("channelMeansTrain.txt", index_col=0, sep="\t")
valX=pd.read_table("channelMeansTrain.txt", index_col=0, sep="\t")
testX=pd.read_table("channelMeansTest.txt", index_col=0, sep="\t")

#Shuffle the dataframes
trainX=trainX.sample(frac=1)
valX=valX.sample(frac=1)
testX=testX.sample(frac=1)   

#Generate binary arrays of classification labels
labelCode={"AutumRoyal":0, "Crimson":1, "Itum4":2, "Itum5":3, "Itum9":4}

trainY=np.zeros(shape=(trainX.shape[0], 5))

for i in range(trainX.shape[0]):
    trainY[i, labelCode[trainX.index[i].split("_")[0]]]=1

valY=np.zeros(shape=(valX.shape[0], 5))
for i in range(valX.shape[0]):
    valY[i, labelCode[valX.index[i].split("_")[0]]]=1

testY=np.zeros(shape=(testX.shape[0], 5))
for i in range(testX.shape[0]):
    testY[i, labelCode[testX.index[i].split("_")[0]]]=1

#%%Initialize 2D-MLP classification model
model=myNeuralNetworks._2D_MLP(dimensions=(37), modelName="simpleMLP", nClasses=5, 
          nUnits=(32,24,16,8))

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

#%%Fit and evaluae normally with the Keras method
mlpHistory=model.fit(x=trainX, y=trainY, batch_size=32, epochs=25,
          validation_data=(valX, valY), class_weight=classWeights)

model.evaluate(x=testX, y=testY, batch_size=32)

#%%History plots & tables
#Plot
N=np.arange(0, 25)
plt.style.use("ggplot")
plt.figure()

for k in ("loss", "val_loss","accuracy","val_accuracy"):
  plt.plot(N, mlpHistory.history[k], label=k)
plt.title("Evolution of loss/metric during training")
plt.xlabel("Epoch nº")
plt.ylabel("Categorical crossentropy/accuracy")
plt.legend()
plt.savefig("lameMLP"+"_History.png")
#Table
header=["loss","val_loss", "accuracy", "val_accuracy"]
histTable=open("lameMLP"+"_modelHistory.txt", "w")
histTable.write("\t".join(header))
histTable.write("\n")

for i in range(25):
  line=[]
  for h in header:
    line.append(str(round(mlpHistory.history[h][i],4)))
  histTable.write("\t".join(line))
  histTable.write("\n")
histTable.close()       
