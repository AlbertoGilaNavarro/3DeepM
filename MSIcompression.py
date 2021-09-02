import numpy as np
import os
import pandas as pd
import cv2 as cv
#Transform 3D MSI arrays into 2D vectors calculating the mean reflectance of every object pixel for every channel

inputDir="mainDir/dataset"
arraysNames=sorted(os.listdir(inputDir))

#Create null array and fill it later
channelMeans=np.zeros((len(arraysNames), 37), dtype=np.float32)

for i in range(len(arraysNames)):
    print(arraysNames[i])
    path=os.path.join(inputDir, arraysNames[i])
    array=np.load(path)
    
    #Filter in the channel 22. This is empirical, you can choose another channel
    mask=np.where(array[:,:,22]>25, 1, 0)
    
    #Saves a visible mask for quality control purposes
    segMask=np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for c in range(3):
        segMask[:,:,c]=np.where(mask==1, 255, 0)
    path=os.path.join("masks", arraysNames[i].split(".")[0]+".png")
    cv.imwrite(path, segMask)
    
    #Extract object pixels. The threshold value is also empirically selected. You can try another value
    index=np.where(array[:,:,22]>25)
    objPixels=array[index[0], index[1], :]
    print(objPixels.shape)
    
    #Calculate mean for every channel and fill in the null array
    for j in range(objPixels.shape[1]):
        m=round(np.mean(objPixels[:,j]),5)
        channelMeans[i,j]=round(np.mean(objPixels[:,j]),5)

#Transform into a pandas dataframe
df_msi=pd.DataFrame(channelMeans)
df_msi.index=arraysNames
colnames=["Channel"+str(i+1) for i in range(37)]
df_msi.columns=colnames

df_msi.to_csv("channelMeans.txt", sep="\t")
