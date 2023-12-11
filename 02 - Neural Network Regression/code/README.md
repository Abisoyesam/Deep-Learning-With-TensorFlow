# Neural Network Regression

- <a href='./01 - NN Regression.ipynb'>NN Regression</a> 
    - Scalar dimension & Vector dimension - Matrix dimension & Tensor

- <a href='./02 - Improving NN Regression.ipynb'>Improving NN Regression Model</a>
  - Improve the model by minimizing the losses to the bearest minimum.
  - Start by creating a simple model with *1 input layer, SGD() optimizer, and epochs of 100* as shown below:
  ```py
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model.compile(losses=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

    # fit the model
    model.fit(X_train, y_train, epochs=100)
  ```
  - You can improve by 2 dense layers (10 neurons maybe), trained for 100 epochs

- <a href='./03 - Evaluating a model.ipynb'>Evaluating the models</a>
    - You can evaluate a model by getting the model summary `model.summary()`
    - You can automatically build the model before gettinf its summary by adding `input_shape=[num_of_shape]` in the input layer 

- <a href='./04 - Evaluation Metrics.ipynb'>Evaluating Linear Regression Metrics</a>
  - Mean Square Error `tf.keras.losses.MSE()`
  - Mean Absolute Error `tf.keras.losses.MAE()`

- <a href='./05 - Comparing models.ipynb'>Comparing Improved Models</a>
  - build different models and tabulation the mse and mae score.
  - tracking models experiments `tensorboard`, `weight and biases`

- <a href='./05 - Comparing models.ipynb'>Saving and Loading Models</a>
  - Saving model `model.save("name of the model")` or `model.save("model_name.h5")`
  - Loading model `tf.keras.models.load_model("fiel path of the model")`
