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
4. Machine learning algorithm or a transfer learning.
5. Output of multiclass or binary classification.

![Input and Output](./images/input_ouput.JPG)

### Input and Output Shapes
- Dimension of the input tensor can be inform of `batch_size`, `width`, `height`, and `colour_channels` 

    ```py
    Shape = [batch_size, width, height, color_channel]
    Shape = [None, 224, 224, 3]

    Shape = [32, 224, 224, 3]
    # 32 is a very common batch size.
    ```
- Output shape is determined by the number of classes. Binary has `2` output as shape while multi class is greater than 2 `>2`.

![Shape](./images/shape.JPG)

> The shape varies depending on the problem you're working with.

## Architecture of the Classification Model
```py
import tensorflow as tf

# 1. Create a model (specified to your problem)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 2. Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# 3. Fit the model
model.fit(X_train, y_train, epochs=5)

# 4. Evaluate the model
model.evaluate(X_test, y_test)
```

<hr>

|Hyperparameter|Binary Classification| Multiclass Classification|
|:-------------|:----:|:------:|
|Input layer shape|Same as number of features|Same as binary classification|
|Hidden layer(s)|Min = 1, Max = Unlimited|Same as binary classification|
|Neurons per hidden layers|generally 10 to 100|Same as binary classification|
|Output layer shape|1 (one class or the other)|1 per class|
|Hidden activation|Usually `ReLU`|Same as binary classification|
|Output activation|Sigmoid|`Softmax`|
|Loss function|Cross entropy `tf.keras.losses.BinaryCrossentropy`|Cross entropy `tf.keras.losses.CategoricalCrossentropy`|
|Optimizer|SDG (stochastic gradient descent), `Adam`|Same as binary classification|