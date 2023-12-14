# Neural Network Classification with TensorFlow

## What are classification problems?
Classification problems are:

- **Binary Classification:** "Is this the thing or not" as in, is this email spam or not; is it a boy or a girl e.t.c
- **Multiclass Classification:** more than one thing or another. E.g Different photos of different animals, different photos of people e.t.c 
- **Multilabel Classification:** Multiple label options per sample.

## Course Outlines
- Architecture of Neural Network classification model
- Input shapes and output shapes of a classification model (features and labels)
- Creating custom data to view and fit
- Steps in modelling
    - Creating a model
    - Compiling a model
    - Fitting a model
    - Evaluating a model
- Different classification evaluation methods
- Saving and loading models

## Classification inputs and outputs
Assuming we are building an food app to classify whether a food is pizza, stake, or shawama?
1. The input is different images of the food.
2. Make sure they are all in the same size (height and width).
3. Change the width, height and color channel (RGB) into tensor by numerically encoding it. 

![Input and Output](./images/input_ouput.JPG)