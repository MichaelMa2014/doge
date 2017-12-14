from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense
from keras.models import Model
import numpy as np
import os
from PIL import Image
import pandas as pd

subm = pd.read_csv('sample_submission.csv')
#labels = pd.read_csv('train/labels.csv')
#labels = labels.set_index('id')

model = ResNet50(weights='imagenet',include_top = True)
#base_model = ResNet50(weights='imagenet',include_top = True, pooling = 'avg')
#predictions = Dense(120, activation='softmax')(base_model.output)
#model = Model(inputs=base_model.input, outputs=predictions)
img_dir = 'test/'
img_list = os.listdir(img_dir)

col = subm.columns
sub = pd.DataFrame(np.zeros([len(img_list),len(col)]),columns=col)
sub['id'] = subm['id']
sub = sub.set_index('id')

cont = 0
for img_path in img_list:
	img = image.load_img(img_dir+img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	pred = decode_predictions(preds)[0]
	#decoder=sorted(decoder,key=lambda item:item[1])
	
	for it in pred:
		lc=it[1].lower()
		if lc in col:
			sub.loc[img_path[0:32],lc] = it[2]
	cont += 1
	if(cont%100==0):
		print(cont)
		print(lc)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)

sub.to_csv('sub_res50_only.csv')
