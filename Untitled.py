#!/usr/bin/env python
# coding: utf-8

# In[2]:




import tensorflow


from keras.applications import ResNet50

img_rows = 265

img_cols = 265 

ResNet50 = ResNet50(weights = 'myimage', 

                 include_top = True, 

                 input_shape = (img_rows, img_cols, 3)

for (i,layer) in enumerate(ResNet50.layers):

    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


from keras.applications import ResNet50
img_rows = 265

img_cols = 265


ResNet50 = ResNet50(weights = 'myimage', 

                 include_top = False, 

                 input_shape = (img_rows, img_cols, 3))





for layer in ResNet50.layers:

    layer.trainable = False





for (i,layer) in enumerate(ResNet50.layers):

    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


def addTopModel(bottom_model, num_class):


    my_model = bottom_model.output

    my_model = GlobalAveragePooling2D()(my_model)

    my_model = Dense(1024,activation='relu')(my_model)
    
    my_model = Dense(1024,activation='relu')(my_model)
    
    my_model = Dense(1024,activation='relu')(my_model)

    my_model = Dense(512,activation='relu')(my_model)

    my_model = Dense(num_classes,activation='softmax')(my_model)

    return my_model



from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

from keras.models import Model



my_class = 4



FC_Head = addTopModel(ResNet50, my_class)



model = Model(inputs=ResNet50.input, outputs=FC_Head)



print(model.summary())





from keras.preprocessing.image import ImageDataGenerator
train_data_dir = './mlops/Data/Train'

validation_data_dir = './mlops/Data/Validation'

train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=20,

      width_shift_range=0.3,

      height_shift_range=0.3,

      horizontal_flip=True,

      fill_mode='nearest')

 

validation_datagen = ImageDataGenerator(rescale=1./255)


train_batchsize = 16

val_batchsize = 19

 

train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_rows, img_cols),

        batch_size=train_batchsize,

        class_mode='categorical')

 

validation_generator = validation_datagen.flow_from_directory(

        validation_data_dir,

        target_size=(img_rows, img_cols),

        batch_size=val_batchsize,

        class_mode='categorical',

        shuffle=False)



from keras.optimizers import RMSprop

 

model.compile(loss = 'categorical_crossentropy',

              optimizer = RMSprop(lr = 0.001),

              metrics = ['accuracy'])

my_train_samples = 120

my_validation_samples = 120

epochs = 8

batch_size = 14



history = model.fit_generator(

    train_generator,

    steps_per_epoch = my_train_samples ,

    epochs = epochs,

    validation_data = validation_generator,

    validation_step = my_validation_sample


from keras.models import load_model



classifier = load_model('my_fam.a1')



import os

import cv2

import numpy as np

from os import listdir

from os.path import isfile, join



Family_dic = {"[0]": "Adil", 

                      "[1]": "an"}



Family_dic_n =  {"n0": "Anam ", 

                         "n1": "an"}



def draw_test(name, pred, im):

    members = Family_dic[str(pred)]

    BLACK = [0,0,0]

    expanded_image = cv2.copyMakeBorder(im, 60, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)

    cv2.putText(expanded_image, members, (10, 50) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)

    cv2.imshow(name, expanded_image)



def getRandomImage(path):

 

    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))

    random_directory = np.random.randint(0,len(folders))

    path_class = folders[random_directory]

    print("Class - " + Family_dict_n[str(path_class)])

    file_path = path + path_class

    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    random_file_index = np.random.randint(0,len(file_names))

    image_name = file_names[random_file_index]

    return cv2.imread(file_path+"/"+image_name)    



for i in range(0,3):

    input_im = getRandomImage("Data/Validation/")

    input_original = input_im.copy()

    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    

    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)

    input_im = input_im.reshape(1,258,258,3) 

    

   
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

    



    
    draw_test("my_fam.a1", res, input_original) 

    cv2.waitKey(0)



cv2.destroyAllWindows()
import cv2
cap = cv2.VideoCapture(0)
while True:
    status,image = cap.read()
    cv2.imshow('Live', image)
    if cv2.waitKey(10)== 13:
        break
cv2.destroyAllWindows()
cap.release()


# In[ ]:




