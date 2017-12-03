import os

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from scipy import ndimage
import numpy as np

from util import TRAIN_PATH, TEST_PATH, OUTPUT_PATH, labels, all_labels, list_images

def image2feature(image):
    """
    Process an image with CNN
    """
    raise NotImplementedError


def create_mode(input_shape, kernel_size=3, pool_size=2):
    model = Sequential()

    # 16 filters
    model.add(Conv2D(16, kernel_size, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(32, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(64, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def train():
    """
    Train CNN
    """
    print("cnn training started")

    model = create_mode((128, 128, 3))
    keras.utils.plot_model(model, to_file=OUTPUT_PATH + "/cnn.png", show_shapes=True)

    dogs = list_images(TRAIN_PATH)[:32]  # TODO Consume huge amount of mem
    print(len(dogs))
    images = []
    ys = []
    for dog in dogs:
        image = ndimage.imread(os.path.join(TRAIN_PATH, dog))
        image = image[:128, :128, :]  # TODO Inconsistent image shapes
        images.append(image)
        label = labels[dog]

        if label is None:
            print("Label not found for train dog %s" % dog)

        y = np.zeros(120)
        y[all_labels.index(label)] = 1
        ys.append(y)
    images = np.array(images)
    ys = np.array(ys)
    model.fit(images, ys, verbose=2)
    model.save(os.path.join(OUTPUT_PATH, "cnn.h5"))


def predict():
    dogs = list_images(TEST_PATH)[:16]  # TODO
    images = []
    for dog in dogs:
        image = ndimage.imread(os.path.join(TEST_PATH, dog))
        image = image[:128, :128, :]  # TODO
        images.append(image)
    images = np.array(images)

    model = keras.models.load_model(os.path.join(OUTPUT_PATH, "cnn.h5"))
    ys = model.predict(images)

    with open(os.path.join(OUTPUT_PATH, "predict.csv"), "w") as out:
        out.write("id, ")
        for label in all_labels:
            out.write(label + ", ")
        out.write("\n")

        for i in range(len(ys)):
            out.write("%s, " % dogs[i])
            for prob in ys[i]:
                out.write("%f, " % prob)
            out.write("\n")


if __name__ == "__main__":
    train()
    predict()
