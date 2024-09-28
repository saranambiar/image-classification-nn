import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Importing libraries:")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.losses import BinaryCrossentropy

img_size = (128,128)
batch_size = 12

print("Applying transformations with ImageDataGenerator:")
new_data = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 25,
    zoom_range = 0.1,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    shear_range = 0.15,
    horizontal_flip = True,
    validation_split = 0.2
)

print("Applying to datasets:")
trained = new_data.flow_from_directory(
    './Stamp_Dataset',
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'binary',
    subset = 'training'
)

validation = new_data.flow_from_directory(
    './Stamp_Dataset',
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'binary',
    subset = 'validation'
)

print("Building neural network:")
tf.random.set_seed(1234)
model = models.Sequential([
    layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128,(3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(1, activation = 'linear')]
)

print("Compiling model:")
model.compile (loss = BinaryCrossentropy(from_logits = True), optimizer = tf.keras.optimizers.Adam(0.001))

steps_per_epoch = int(np.ceil(trained.samples / batch_size))
validation_steps = int(np.ceil(validation.samples / batch_size))

print("Fitting model:")
history = model.fit(
    trained,                     
    epochs=60,                      
    validation_data = validation, 
)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

print("Plotting model loss:")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print(model.summary())