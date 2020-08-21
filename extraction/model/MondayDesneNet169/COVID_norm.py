#!/usr/bin/env python
# coding: utf-8

# We have a set of images for COVID and other folder for Not COVID. Another folder named split-data contains files that represent the distribution of the images into training validation and testing. 

# In[1]:


from sklearn.metrics import cohen_kappa_score, accuracy_score


# In[2]:


import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adadelta,Adamax
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
#from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy
from keras.applications.resnet50 import preprocess_input
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


# In[31]:


BATCH_SIZE=8


# In[3]:



themodel='DenseNet169'
IMG_SIZE=224
historyfilename='densenet121/history_model_norm.json'
modelweights='densenet121/model_weights_norm.h5'
filetosaveto="densenet121/extractedfeatures.pickle"


# ## Data Collection Process

# In[4]:


#The image names are found in the data-split folder
filename_train_covid='Data-split/COVID/trainCT_COVID.txt'
filename_val_covid='Data-split/COVID/valCT_COVID.txt'
filename_test_covid='Data-split/COVID/testCT_COVID.txt'

filename_train_no_covid='Data-split/NonCOVID/trainCT_NonCOVID.txt'
filename_val_no_covid='Data-split/NonCOVID/valCT_NonCOVID.txt'
filename_test_no_covid='Data-split/NonCOVID/testCT_NonCOVID.txt'


# In[5]:


path_covid='Images-processed/CT_COVID/CT_COVID/'
path_non_covid='Images-processed/CT_NonCOVID/CT_NonCOVID/'


# In[6]:



#prepare training data
with open(filename_train_covid) as f:
    covidimages_train_path = f.read().splitlines()

covidimages_train_path=[path_covid+s for s in covidimages_train_path]
covidclasses=[1]*len(covidimages_train_path)
print("Number of COVID images in train dataset is :"+ str(len(covidclasses)))

with open(filename_train_no_covid) as f:
    no_covidimages_train_path = f.read().splitlines()
non_covidimages_train_path=[path_non_covid+s for s in no_covidimages_train_path]
non_covidclasses=[0]*len(non_covidimages_train_path)
print("Number of Non COVID images in train dataset is :"+ str(len(non_covidclasses)))

trainpaths=covidimages_train_path+non_covidimages_train_path
Y_train=covidclasses+non_covidclasses

print("training samples :"+ str(len(Y_train)))


# In[7]:



#prepare val data
with open(filename_val_covid) as f:
    covidimages_val_path = f.read().splitlines()

covidimages_val_path=[path_covid+s for s in covidimages_val_path]
covidclasses=[1]*len(covidimages_val_path)
print("Number of COVID images in val dataset is :"+ str(len(covidclasses)))

with open(filename_val_no_covid) as f:
    no_covidimages_val_path = f.read().splitlines()
non_covidimages_val_path=[path_non_covid+s for s in no_covidimages_val_path]
non_covidclasses=[0]*len(non_covidimages_val_path)
print("Number of Non COVID images in val dataset is :"+ str(len(non_covidclasses)))

valpaths=covidimages_val_path+non_covidimages_val_path
Y_val=covidclasses+non_covidclasses

print("valing samples :"+ str(len(Y_val)))


# In[8]:



#prepare test data
with open(filename_test_covid) as f:
    covidimages_test_path = f.read().splitlines()

covidimages_test_path=[path_covid+s for s in covidimages_test_path]
covidclasses=[1]*len(covidimages_test_path)
print("Number of COVID images in test dataset is :"+ str(len(covidclasses)))

with open(filename_test_no_covid) as f:
    no_covidimages_test_path = f.read().splitlines()
non_covidimages_test_path=[path_non_covid+s for s in no_covidimages_test_path]
non_covidclasses=[0]*len(non_covidimages_test_path)
print("Number of Non COVID images in test dataset is :"+ str(len(non_covidclasses)))

testpaths=covidimages_test_path+non_covidimages_test_path
Y_test=covidclasses+non_covidclasses

print("testing samples :"+ str(len(Y_test)))


# In[9]:




def equalize_light(image, limit=2, grid=(16,16), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x) #typically will be zero
    max_val = np.max(x) #typically will be 255
    x = (x-min_val) / (max_val-min_val)
    return x
def load_ben_color(path, sigmaX=10 ):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = equalize_light(image,3,(5,5))
    #image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    image=normalize(image)  
    return image






N = len(trainpaths)
X_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)

for i, image_path in enumerate(tqdm(trainpaths)):
    X_train[i, :, :, :] = load_ben_color(image_path,sigmaX=10)


# In[14]:


print(X_train.shape)


# In[15]:


N_val = len(valpaths)
X_val = np.empty((N_val, IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)

for i, image_path in enumerate(tqdm(valpaths)):
    X_val[i, :, :, :] = load_ben_color(image_path,sigmaX=10)
print(X_val.shape)


# In[16]:


N_test = len(testpaths)
X_test = np.empty((N_test, IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)

for i, image_path in enumerate(tqdm(testpaths)):
    X_test[i, :, :, :] = load_ben_color(image_path,sigmaX=10)
print(X_test.shape)


# ## Setup data generators

# In[17]:


class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                print("value is :")
                print(y_train_)
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


# In[18]:



def create_datagen():
    return ImageDataGenerator(
        #rescale=1./255,
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(X_train, Y_train, batch_size=BATCH_SIZE)


# ## Metrics Function

# In[19]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_pred = self.model.predict(X_val)
        y_pred=np.round(y_pred)

        _val_kappa = cohen_kappa_score(
            y_val.argmax(axis=1), 
            y_pred.argmax(axis=1), 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        return


# ## Pre-Trained

# In[20]:


import keras
import numpy as np

keras.backend.clear_session()

  
def change_model(model, new_input_shape=(None, 40, 40, 3)):
    # replace input shape of first layer
    model._layers[0].batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            #print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model
  


# In[21]:





from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input,decode_predictions
if themodel=='DenseNet121':
    from keras.applications import DenseNet121
    if IMG_SIZE !=224:
        model = DenseNet121(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = DenseNet121(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel=='DenseNet169':
    from keras.applications import DenseNet169
    if IMG_SIZE !=224:
        model = DenseNet169(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = DenseNet169(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel=='InceptionV3':
    from keras.applications import InceptionV3
    if IMG_SIZE !=224:
        model = InceptionV3(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel=='InceptionResNetV2':
    from keras.applications import InceptionResNetV2
    if IMG_SIZE !=224:
        model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel=='DenseNet201':
    from keras.applications import DenseNet201
    if IMG_SIZE !=224:
        model = DenseNet201(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = DenseNet201(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel =='Xception':
    from keras.applications.xception import Xception
    if IMG_SIZE !=224:
        model = Xception(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = Xception(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel =='MobileNet':
    from keras.applications.mobilenet import MobileNet
    if IMG_SIZE !=224:
        model = MobileNet(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = MobileNet(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel =='MobileNetV2':
    from keras.applications import MobileNetV2
    if IMG_SIZE !=224:
        model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224,3))
elif themodel =='ResNet50':
    from keras.applications import ResNet50
    if IMG_SIZE !=224:
        model = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224,3))
        new_model = change_model(model,new_input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    else:
        new_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224,3))
    


new_model.summary()


# In[26]:


def build_model():
    model = Sequential()
    model.add(new_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(13, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00004),#Adam(lr=0.00005)
        metrics=['accuracy']
    )
    
    return model


# In[27]:


model = build_model()


# In[28]:


Y_train= np.array(Y_train)
Y_train=Y_train.reshape(-1,1)
Y_val= np.array(Y_val)
Y_val=Y_val.reshape(-1,1)


# In[29]:


print(X_train.shape)
print(Y_train.shape)


# In[ ]:


kappa_metrics = Metrics()

checkpoint = ModelCheckpoint(modelweights, 
    monitor='val_loss', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto',patience=7)
#history = model.fit(
#    X_train,Y_train,
#    epochs=5,
#    batch_size=BATCH_SIZE,
#    validation_data=(X_val, Y_val),
#    callbacks=[checkpoint,kappa_metrics,early_stopping]
#)
history = model.fit_generator(
    data_generator,
    steps_per_epoch=X_train.shape[0] / BATCH_SIZE,
    epochs=50,
    validation_data=(X_val, Y_val),
    callbacks=[checkpoint,early_stopping]
)


# In[ ]:


with open(historyfilename, 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df.to_csv('history.csv')
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()



print("Preds after loading weights")
model.load_weights(modelweights)
preds = model.predict(X_test, verbose=2)
from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc
print(accuracy_score(Y_test,np.round(preds)))
print(f1_score(Y_test,np.round(preds)))
fpr, tpr, thresholds = roc_curve(Y_test, preds)
print("AUC: " + str(auc(fpr, tpr)))

from keras.models import Model
layer_name = 'global_average_pooling2d_1'#'global_average_pooling2d_1'
model_extracted= Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
model_extracted.summary()

Train_Features=model_extracted.predict(X_train)
Train_Target=Y_train

Val_Features=model_extracted.predict(X_val)
Val_Target=Y_val

Test_Features=model_extracted.predict(X_test)
Test_Target=Y_test

import pickle

with open(filetosaveto, "wb") as f:
	pickle.dump((Train_Features,Train_Target,Val_Features,Val_Target,Test_Features,Test_Target,model,model_extracted), f)




