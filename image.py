
# import the necessary packages
import numpy as np
import time
import urllib
import cv2
import csv 
import cv2
import numpy as np
import numpy as np
#import urllib
import cv2
import urllib.request
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import merge, Input
import os

from urllib.request import urlopen
from urllib.request import urlretrieve
from operator import itemgetter
# import ctypes
# libc = ctypes.cdll.LoadLibrary('libc.so.6')
# res_init = libc.__res_init
# # ...
# res_init()

# import os
# os.environ['http_proxy']=''
image_input = Input(shape=(224,224,3))
model_vgg = VGG16(include_top=False,weights="imagenet",input_tensor=image_input)
# model.summary()
 
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	os.environ['http_proxy'] = ''
	try:
		req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
		html = (urllib.request.urlopen(req).read())
		img_array = np.array(bytearray(html))
		im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	 
		# return the image
		return im
	except:
		para="no"
		return para
count=0
list_urls=[]
list_Overallsentiment=[]
images=[]
with open('data7000.csv') as myFile:  
	reader = csv.DictReader(myFile)
	for row in reader:
		list_urls.append((row['Image_URL']))
		list_Overallsentiment.append((row['Overall_Sentiment']))
        #list_name.
		count=count+1
		if(count>50):
			break
print(len(list_urls))
print(len(list_Overallsentiment))
count=0
indexes=set()
# x=[]
# i=0
# loop over the image URLs
for url in list_urls:
	# download the image URL and display it
	#print "downloading %s" % (url)
	im = url_to_image(url)
	x=list_urls.index(url)
	print("index",x)
	#x.append()
# 	# y=list_Overallsentiment.index(url)
# 	# print("index",y)
	if im=="no":
		print("not found")
		list_urls.remove(list_urls[x])
		list_Overallsentiment.remove(list_Overallsentiment[x])
		indexes.add(x)

	# else:
	# 	images.append(im)
for url in list_urls:
	
	im = url_to_image(url)
	images.append(im)

print("After remove url",len(list_urls))
print(len(list_Overallsentiment))
print(len(images))

	# cv2.imshow("Image", image)
	# cv2.waitKey(0)

# print((images))
# cv2.imshow("Image", images[0])
# cv2.waitKey(0)

# print(len(images))
vgg16_feature_list = []

#for idx, dirname in enumerate(subdir):
    # get the directory names, i.e., 'dogs' or 'cats'
    # ...
#[[for x in range(len(images))] for x in range(len(images)) for x in range(len(images))   
count=0
for i in range(len(images)):
        # process the files under the directory 'dogs' or 'cats'
        # ...
        
	# img = image.load_img(images[i], target_size=(224, 224))
	# img_data = .img_to_array(img)
	im = cv2.resize(images[i], (224, 224)).astype(np.float32)
	count=count+1
	#imgUMat = cv2.UMat(im)
	#im=cv2.cvtColor(cv2.UMat(imUMat), cv2.COLOR_RGB2GRAY)
	img_data = np.expand_dims(im, axis=0)
	img_data = preprocess_input(img_data)
	vgg16_feature = model_vgg.predict(img_data)
	vgg16_feature_np = np.array(vgg16_feature)
	vgg16_feature_list.append(vgg16_feature_np.flatten())
print("count",count)
print("fet list :",len(vgg16_feature_list))
vgg16_feature_list_np = np.array(vgg16_feature_list)
# print(vgg16_feature_list_np)
print("vgg shape",vgg16_feature_list_np.shape)
import pandas as pd 
import numpy as np
from numpy import array
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
# results = [int(i) for i in list_Overallsentiment]
# print(results)
# le=LabelEncoder()
# Y=le.fit_transform(list_Overallsenitment)
# print(Y)
values=array(list_Overallsentiment)
print(values)
label_encoder=LabelEncoder()
integer_encoded=label_encoder.fit_transform(values)
print(integer_encoded)
from keras.utils import to_categorical
integer_encoded=to_categorical(integer_encoded)
print(integer_encoded)
print("integer_encoded shape",integer_encoded.shape)
X_train, X_test, y_train, y_test = train_test_split(vgg16_feature_list_np,integer_encoded,test_size=0.1)
import keras
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
epochs = 100
model = models.Sequential()
model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)
model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)
model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)
model.add(layers.Dense(5, activation='sigmoid'))
model.summary()
# Compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
