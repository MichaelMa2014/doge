from PIL import Image
import numpy as np
import os
import keras

size = 128
def preprocessing(path):
	images = []
	dir = os.listdir(path)
	cont = 0
	for ipath in dir:
		img = Image.open(path+ipath)
		img = img.resize([size,size])
		images.append(np.array(img))
		cont += 1
		if cont==100:
			break
	return images
	
if __name__ == "__main__":
	x_train = preprocessing('train\\')
	datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
    samplewise_center=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True)
	
	datagen.fit(x_train)
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
	