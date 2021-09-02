import numpy as np
import pandas as pd
import os
import random
import albumentations as A
import cv2 as cv
import matplotlib.pyplot as plt

import myNeuralNetworks

#DataAugmentation generator. Intented to be used only in classification problems, not regression ones
def generator(inputDir, batchSize, augment=False, randomSeed=42):
    #Controls reproducibility
    np.random.seed(seed=randomSeed)
    random.seed(randomSeed)
    
    #Define pipeline of DA, this is hard-coded
    augmentator=A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1)
        ], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.01,
            scale_limit=0.3,
            rotate_limit=90,
            interpolation=cv.INTER_LINEAR,
            border_mode=cv.BORDER_CONSTANT,
            value=10,
            p=0.5)])
    
    #Define classification labels
    labelCode={"AutumRoyal":0, "Crimson":1, "Itum4":2, "Itum5":3, "Itum9":4}
    
    #Get filename
    arraysNames=os.listdir(inputDir)
    
    #Used to load the whole dataset without DA
    loadAux=0
    
    while (True):
        if augment==False:
            Index=np.random.choice(range(len(arraysNames)), size=batchSize,
                                   replace=False)
        else:
            #Create batches sequentially. 
            if (loadAux+batchSize < len(arraysNames)-1):
                Index=list(range(loadAux, loadAux+batchSize))
                loadAux+=batchSize
            else:
                Index=list(range(loadAux, len(arraysNames)))
                loadAux=0 
                
        #Crea null array and fill later 
        batchArray=np.zeros(shape=(batchSize,140,200,37), dtype=np.float16)
        labelsArray=np.zeros(shape=(batchSize, 5), dtype=np.uint8)
        
        #Load=>Transforms (or not)=>Saves in array       
        for (n,i) in enumerate(Index): 
            image=np.load(os.path.join(inputDir, arraysNames[i]))
            if augment==True:
                image=augmentator(image)["image"]
            batchArray[n]=image
            labelsArray[n, labelCode[arraysNames[i].split("_")[0]]]=1
        
        #Normalize image array dividing by 255
        batchArray=batchArray/255
        
        yield (batchArray, labelsArray)
        
#%% Input parameters
#You must split the dataset into train, validation and test subsets manually, respecting the class proportions
trainDir="subDataSets/Train" #Ajuste de parametros entrenables
valDir=trainDir.replace("Train", "Val") #Seleccion de modelos
testDir=trainDir.replace("Train", "Test") #Evaluacion final

epochNumber=10
batchSize=16
trainSteps=int(len(os.listdir(trainDir))/batchSize)
valSteps=int(len(os.listdir(valDir))/batchSize)
testSteps=int(len(os.listdir(testDir))/batchSize)

modelArgs={"dimensions":(140,200,37,1), "modelName":"3DeepM_KS5-10",
           "nClasses":5, "kernelSizes":((5,5,5),(10,10,10))}
classWeights={0:2.5, 1:1.25, 2:6, 3:1, 4:5.25}

#%% Initialize generators
trainGenerator=generator(inputDir=trainDir, batchSize=batchSize, augment=True)
valGenerator=generator(inputDir=valDir, batchSize=batchSize, augment=True)
testGenerator=generator(inputDir=testDir, batchSize=batchSize, augment=False)

#%% Initialize a 3D-CNN classification model
model=myNeuralNetworks._3D_CNN(**modelArgs)
                
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

#%% Keras does not re-initialize a custom generator after each epoch automatically.
#It must be done manually to ensure that the same augmented images are used in every epoch
trainHistory={"loss":[], "val_loss":[], "accuracy":[], "val_accuracy":[]}
for e in range(epochNumber):
    print("Epoch: ",(e+1), "/",epochNumber)
    trainGenerator=generator(inputDir=trainDir, batchSize=16, augment=False)
    valGenerator=generator(inputDir=valDir, batchSize=16, augment=False)
    hist=model.fit(x=trainGenerator, epochs=1, validation_data=valGenerator,
                   steps_per_epoch=300, validation_steps=100,
                   class_weight=classWeights)
    for k in trainHistory.keys():
        trainHistory[k].append(hist.history[k][0])
 
#%% Validation with built-in Keras function
model.evaluate(x=testGenerator, batch_size=16, steps=testSteps)
#100% accuracy. Experiment succesfully replicated

#%%History plots & table
#Plot
N=np.arange(0, epochNumber)
plt.style.use("ggplot")
plt.figure()

for k in ("loss", "val_loss","accuracy","val_accuracy"):
  plt.plot(N, trainHistory[k], label=k)
plt.title("Evolution of loss/metric during training")
plt.xlabel("Epoch nº")
plt.ylabel("Categorical crossentropy/accuracy")
plt.legend()
plt.savefig(modelArgs["modelName"]+"_History.png")

#Table
header=["loss","val_loss", "accuracy", "val_accuracy"]
histTable=open(modelArgs["modelName"]+"_modelHistory.txt", "w")
histTable.write("\t".join(header))
histTable.write("\n")

for i in range(epochNumber):
  line=[]
  for h in header:
    line.append(str(round(trainHistory[h][i],4)))
  histTable.write("\t".join(line))
  histTable.write("\n")
histTable.close()        



