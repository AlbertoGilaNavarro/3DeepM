def _3D_CNN(dimensions,modelName, nClasses, kernelSizes,
                         multiOutput=False):
  inputLayer=Input(shape=dimensions)

  x=Conv3D(filters=8, kernel_size=kernelSizes[0], padding="same", dilation_rate=(2,2,2),
           kernel_initializer="he_uniform",
           name="Conv3D_KS"+str(kernelSizes[0][0]))(inputLayer)
  x=Activation("relu", name="ReLU_Conv1")(x)
  x=BatchNormalization(name="BatchNorm_Conv1")(x)
  
  x=AveragePooling3D((2,2,2), name="AvgPool2x2x2")(x)

  x=Conv3D(filters=16, kernel_size=kernelSizes[1],padding="same", dilation_rate=(2,2,2),
           kernel_initializer="he_uniform",name="Conv3D_KS"+str(kernelSizes[1][0]))(x)
  x=Activation("relu", name="ReLU_Conv2")(x)
  x=BatchNormalization(name="BatchNorm_Conv2")(x)

  x=GlobalAveragePooling3D(name="GlobalAvgPool3D")(x)

  if(multiOutput==False):

    x=Dense(units=16, activation="relu", name="HiddenLayer_16N")(x)
    finalLayer=Dense(units=nClasses, activation="softmax",name="OutputLayer_5N")(x)
    
    model=Model(inputs=inputLayer, outputs=finalLayer, name=modelName)
  
  else:
    dense0=Dense(units=16, activation="relu", name="HiddenLayer0_16N")(x)
    itum9_layer=Dense(units=1, activation="sigmoid", name="Itum9")(dense0)
    
    dense1=Dense(units=16, activation="relu", name="HiddenLayer1_16N")(x)
    crimson_layer=Dense(units=1, activation="sigmoid", name="Crimson")(dense1)

    dense2=Dense(units=16, activation="relu", name="HiddenLayer2_16N")(x)
    itum4_layer=Dense(units=1, activation="sigmoid", name="Itum4")(dense2)

    dense3=Dense(units=16, activation="relu", name="HiddenLayer3_16N")(x)
    itum5_layer=Dense(units=1, activation="sigmoid", name="Itum5")(dense3)

    dense4=Dense(units=16, activation="relu", name="HiddenLayer4_16N")(x)
    autumRoyal_layer=Dense(units=1, activation="sigmoid", name="AutumRoyal")(dense4)

    model=Model(inputs=inputLayer, name=modelName,
                outputs=[itum9_layer, crimson_layer, itum4_layer, itum5_layer, autumRoyal_layer])
  
  model.summary()
  return model

def _2D_MLP(dimensions, modelName, nClasses, nUnits):
    inputLayer=Input(shape=dimensions, name="Input")
    x=Dense(units=nUnits[0], activation="relu",
            kernel_initializer="he_uniform", name="Dense1")(inputLayer)
    x=BatchNormalization(name="BNorm1")(x)
    x=Dense(units=nUnits[1], activation="relu",
            kernel_initializer="he_uniform", name="Dense2")(x)
    x=BatchNormalization(name="BNorm2")(x)
    x=Dense(units=nUnits[2], activation="relu",
            kernel_initializer="he_uniform", name="Dense3")(x)
    x=BatchNormalization(name="BNorm3")(x)
    x=Dense(units=nUnits[3], activation="relu",
            kernel_initializer="he_uniform", name="Dense4")(x)
    x=BatchNormalization(name="BNorm4")(x)
    outputLayer=Dense(units=nClasses, activation="sigmoid", name="Output")(x)
    
    model=Model(inputs=inputLayer, outputs=outputLayer, name=modelName)
    model.summary()
    return model



#Example of usage
#3D
cnnArgs={"dimensions":(140,200,37,1), "modelName":"3DeepM_KS5-10",
           "nClasses":5, "kernelSizes":((5,5,5),(10,10,10))}
model=_3D_CNN(**cnnArgs)
#2D
mlpArgs={"dimensions":37, "modelName":"simpleMLP",
           "nClasses":5, "nUnits":(32,24,16,8)}
model=_2D_MLP(**mlpArgs)

