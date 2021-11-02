# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:35:48 2020

@author: Armando Fernandes
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

IMAGE_DIMENSION = 128
BS = 32
EPOCHS = 50

print(' --------------------Pré-processamento das imagens--------------------')

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape = (IMAGE_DIMENSION, IMAGE_DIMENSION, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (4,4)))

model.add(Conv2D(32, (3,3), input_shape = (IMAGE_DIMENSION, IMAGE_DIMENSION, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

print(' --------------------Rede neural densa--------------------')

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation = 'sigmoid'))


print('\n---------------------Compilação--------------------\n')


model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])


'''
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
'''


model.summary()


print('\n---------------------Data augmentation--------------------\n')


gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2
                                         )
gerador_teste = ImageDataGenerator(rescale = 1./255)


base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (IMAGE_DIMENSION, IMAGE_DIMENSION),
                                                           batch_size = BS,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (IMAGE_DIMENSION, IMAGE_DIMENSION),
                                               batch_size = BS,
                                               class_mode = 'binary')



print('\n---------------------Treinamento--------------------\n')



#checkpoint = ModelCheckpoint("pesos.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min')


history = model.fit_generator(base_treinamento, steps_per_epoch = 1000/BS,
                            epochs = EPOCHS, validation_data = base_teste,
                            validation_steps = 1000/BS, callbacks=[reduce_lr])





print('\n---------------------Validação--------------------\n')


imagem_teste = image.load_img('dataset/HMIIF.jpg',
                              target_size = (IMAGE_DIMENSION,IMAGE_DIMENSION))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = model.predict(imagem_teste)

#previsao = (previsao > 0.5)

base_treinamento.class_indices


'''
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["acc"], label="train_acc")
plt.plot(N, history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Época #")
plt.ylabel("Loss/Accuracy")
plt.legend()
'''

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
