# Neural Network Classification

- <a href='./01 -  NN Classification.ipynb'>NN Classification</a> 
    - Make use of `sklearn` demo data called `make_circle`
    - Use `accuracy` as metrics and result is converted to percentage. 
    - 50% accuracy means that the **model is guessing**.
    - `plot_decision_boundary()` function

- <a href='./02 - Non-linearity.ipynb'>NN Classification Non Linearity</a> 
    - Try out tensorflow playground
    - Change activation to `non-linear`
    - Replicate `Sigmoid` function
    - Visualizing history to plot model loss curves
    - Finding best learning rate using `callbacks`

- <a href='./03 -  NN Classification Evaluation.ipynb'>Classification Evaluation Methods</a> 
    - `accuracy`, `precision`, `confusion matrix`, `recall`, `f1-score`
    - Anatomy of confusion matrix
    - `False negative`, `False positive`, `True positive`, `True negative`.

- <a href='./04 - Multiclass Classification.ipynb'>Multi Class Neural Network Classification</a> 
    - Loading fashion mnist from `tf.keras.datasets`
    - Getting familiar with the data
    - `Flatten()` input shapes
    - Losses for multiclass classification can be either `CategoricalCrossentropy` or `SparseCategoricalCrossentropy`.
    - Normalizing data (range 0 to 1); divide by `255`
    - Finding ideal learning rate