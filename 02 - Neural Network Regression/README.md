# Neural Network Regression with TensorFlow

## What is Regression Problem?
These are problems trying to predict a continous value (number) such as house price, cost of health insurance and so on.

![Regression Problem](./images/reg_prob.JPG)

**N.B:** Notice the word *how much* or *how many*. These are most often pointer of regression problems. Another sort of regression problem is trying to predict the location of bonding box in a classification problem (object detection).

![Class_Reg](./images/bb.JPG)

## Course Outline
- Architecture of a neural network regression model
- Input shapes and output shape of a regression model (features and labels)
- Creating custom data to view and fit
- Steps in modelling
    - Creating a model, compiling a model, fitting a model, evaluating a model
- Different evaluation methods

## Regression Inputs and Output

![Input_Output](./images/input_output.JPG)

The inputs are features whose shape could be more than one but the output is just a single outcome.

## Architecture of a regression model

|Hyperparameters| Typical Value|
|:--------------:|:-------------|
|Input layer shape|Same shape as the number of features (e.g **3** for bedrooms, bathroom, and car spaces in housing price prediction)|
|Hidden layer (s)|It is problem specific. min = 1, max = unlimited|
|Neurons per hidden layer|It is problem specific, generally **10 to 100**.|
|Output layer shape|Same shape as desired prediction shape (e.g 1 for house price)|
|Hidden activation|Usually **RELU** (rectified linear unit)|
|Output activation|None, RELU, logistic/tanh|
|Loss function|**MSE** (Mean Square Error) or **MAE** (Mean Absolute Error) or Huber (combination of MAE & MSE) if outliers|
|Optimizer|**SDG** (stochastic gradient descent), **Adam**|