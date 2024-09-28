# Training a neural network on a custom dataset:
# Task:
Classify documents on the basis of presence of a 100 rupee stamp.

Dataset - collected from Google Images.
File structure is as follows:
Stamp_Model
- stamp (contains images with 100 rupee stamp)
- nostamp (contains images with no stamps/ stamps other than requirement)

# Data augmentation:
Due to a smaller size dataset, we augment the images using ImageDataGenerator from Keras.

# Creating the neural network:
- We implement Conv2D and MaxPooling2D layers, along with 3 Dense layers using 'relu' and 'linear' activation functions.
- Loss: BinaryCrossentropy (for binary classification)
- Optimizer: Adaptive Moment Estimation
- Learning rate: 0.001

