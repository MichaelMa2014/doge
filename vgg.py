import os

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import keras.preprocessing.image
import numpy as np

from util import TRAIN_PATH, TEST_PATH, OUTPUT_PATH, labels, all_labels, list_images

width = 224
height = 224


def image2feature_vgg(path):
    img = keras.preprocessing.image.load_img(path, target_size=(width, height))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    model = create_model((3, width, height))
    return model.predict(x)


def create_model(input_shape=(3, width, height)):
    vgg_model = VGG16(weights='imagenet')
    flatten = vgg_model.get_layer("flatten")

    model = Sequential()
    model.add(flatten)
    model.add(Dense(10000))
    model.add(Dense(120, activation="softmax", input_shape=input_shape))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train():
    model = create_model()
    dogs = list_images(TRAIN_PATH)[:10]
    images = []
    ys = []
    for dog in dogs:
        image = keras.preprocessing.image.load_img(os.path.join(TRAIN_PATH, dog), target_size=(width, height))
        image = keras.preprocessing.image.img_to_array(image)
        images.append(image)
        label = labels[dog]

        if label is None:
            print("Label not found for train dog %s" % dog)

        y = np.zeros(120)
        y[all_labels.index(label)] = 1
        ys.append(y)
    images = np.array(images)
    ys = np.array(ys)
    model.fit(images, ys, verbose=2, epochs=1)
    model.save(os.path.join(OUTPUT_PATH, "vgg.h5"))


def predict():
    dogs = list_images(TEST_PATH)[:10]  # TODO
    images = []
    for dog in dogs:
        image = keras.preprocessing.image.load_img(os.path.join(TEST_PATH, dog), target_size=(width, height))
        image = keras.preprocessing.image.img_to_array(image)
        images.append(image)
    images = np.array(images)

    model = keras.models.load_model(os.path.join(OUTPUT_PATH, "vgg.h5"))
    ys = model.predict(images)

    with open(os.path.join(OUTPUT_PATH, "predict_vgg.csv"), "w") as out:
        out.write("id,")
        for label in all_labels[:-1]:
            out.write(label + ",")
        out.write(all_labels[-1])
        out.write("\n")

        for i in range(len(ys)):
            out.write("%s," % dogs[i].split(".")[0])
            for prob in ys[i][:-1]:
                out.write("%f," % prob)
            out.write("%f" % ys[i][-1])
            out.write("\n")


if __name__ == "__main__":
    train()
    predict()
