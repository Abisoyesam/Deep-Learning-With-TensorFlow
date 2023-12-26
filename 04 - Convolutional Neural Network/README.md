# Convolutional Neural Network and Computer Vision

## What is a Computer Vision Problem?
Computer being used for visual problem. Training the computer to see. It applications are: 
- Self driving cars
- Object detection

## What to cover?
- Getting pictorial dataset to work with **pizza & steak** :pizza: 
- Architecture of Convolutional Neural Network (CNN) with TensorFlow.
- An end-to-end binary image classification problem.
- Steps in modelling with CNNs
    - Creating CNN, Compiling a model, Fitting a model, Evaluating a model.
- An end-to-end multi-class image classification problem.
- Making prediction over custom images.

## Computer Vision Input and Output
![input_output](./images/input.JPG)

## Architecture of CNN

```py
model.keras.Sequential([
    tf.keras.layer.Conv2D(filters=10, 
        kernel_size=3, # can also be (3,3)
        activation="relu",
        input_shape=(224, 224, 3)),
        # height width, color channel
    tf.keras.layers.Conv2D(10, 3,
        activation="relu"),
    
])
```