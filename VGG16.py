import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import VGG16
from keras.optimizers import Adam
import keras.backend as K
K.set_image_dim_ordering('tf')

#Generating training and testing datasets
train_datagen= ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.1, horizontal_flip= True)
valid_datagen= ImageDataGenerator(rescale=1./255)
size=(128,128)
in_shape=(128,128,3)
train_set= train_datagen.flow_from_directory('Final/train', 
                                             target_size=size, batch_size=100, class_mode='categorical', 
                                             shuffle=True, seed=20)
valid_set= valid_datagen.flow_from_directory('Final/val', 
                                             target_size=size, batch_size=100, class_mode='categorical', 
                                             shuffle=False)

#using pre-trained model
base_model=VGG16(input_shape=in_shape, weights='imagenet', include_top=False)

x=base_model.output
x=Conv2D(32, (3,3), activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Flatten()(x)
x=Dense(units=128, activation='relu')(x)
x=Dense(units=64, activation='relu')(x)
x=Dense(units=32, activation='relu')(x)
x=Dense(units=14916, activation='softmax')(x)

model=Model(inputs=base_model.inputs, outputs=x)

for layer in model.layers[:16]:
  layer.trainable=False

for layer in model.layers[16:]:
    layer.trainable=True

#Compile and fit the datasets
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
step_size_train=train_set.n//train_set.batch_size
step_size_valid=valid_set.n//valid_set.batch_size
model.fit_generator(train_set, steps_per_epoch=step_size_train, epochs=60, 
                    validation_data= valid_set, validation_steps=step_size_valid)

model.save('save_model.h5')

#Predict for new images
import os
label=os.listdir('Final/test')
pred=np.array([])
conf=np.array([])
true=np.array([])
i=0
for a in label:
	for st in os.listdir('Final/test/'+a):
		img=image.load_img(('Final/test/'+a+'/')+st, target_size=size)
		img=image.img_to_array(img)
		img=img.reshape(1,128,128,3)
		output=model.predict(img)
		pred=np.append(pred, (np.argmax(output[0])))
		conf=np.append(conf, np.amax(output[0]))
		true=np.append(true, int(a))
		i+=1

#Find and print GAP score
def GAP(pred, conf, true, return_x=False):
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap

gap=GAP(pred, conf, true)
print('GAP score: ',gap)

