from PIL import Image
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Activation,Conv2D,Dense,Flatten
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator

def preprocessing(path,size = 128):
    imgpath = path+'images//'
    labelcsv=pd.read_csv(path+'labels.csv')
    labelcsv = labelcsv.set_index('id')
    
    images = []
    labels = []
    dir = os.listdir(imgpath)
    for ipath in dir:
        img = Image.open(imgpath+ipath)
        img = img.resize([size,size])
        images.append(np.array(img))
        labels.append(labelcsv.loc[ipath[0:32],'breed'])
    images = np.array(images)
    classes = len(set(labels))
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels).reshape([len(labels),1])
    ohe = OneHotEncoder()
    ohe.fit(labels)
    labels = np.array(ohe.transform(labels).toarray())
    return images,labels,classes

def create_model(input_shape,classes):
    return model
    
if __name__ == "__main__":
    
    x_train,y_train,classes = preprocessing('train//')

    datagen = ImageDataGenerator(featurewise_center=True,
    samplewise_center=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True)
    
    datagen.fit(x_train)
    model = create_model((128,128,3),classes)

    datagen.flow(x_train, y_train, batch_size=32)
    epochs = 10
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
    for e in range(epochs):
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
                break
    model.fit(x_train, y_train, verbose=2)
