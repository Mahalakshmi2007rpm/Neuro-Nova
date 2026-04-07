import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

def unet():
    inputs = layers.Input((224,224,3))

    c1 = layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32,3,activation='relu',padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64,3,activation='relu',padding='same')(p2)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1,c2])

    u2 = layers.UpSampling2D()(u1)
    u2 = layers.concatenate([u2,c1])

    outputs = layers.Conv2D(1,1,activation='sigmoid')(u2)

    return Model(inputs, outputs)

model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Dummy data (replace later)
X = np.random.rand(10,224,224,3)
Y = np.random.rand(10,224,224,1)

model.fit(X,Y,epochs=3)

os.makedirs('models', exist_ok=True)
model.save('models/unet_model.h5')
print('Saved models/unet_model.h5')