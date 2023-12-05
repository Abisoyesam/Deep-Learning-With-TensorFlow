# Fundamental of TensorFlow -  Code Outlines

- <a href='01 - Tensorflow.ipynb'>Creating Tensor with `tf.constant`</a>
    - Scalar dimension & Vector dimension
    - Matrix dimension & Tensor 

- <a href='./02 - Creating Tensor.ipynb'>Creating Tensor with `tf.Variable`</a>
    - `.assign()` object

- <a href='./03 - Random & Shuffling Tensor.ipynb'>Creating Random Tensor and Shuffling Tensor</a>
    - Random tensor from uniform and normal distribution
    - Shuffling tensor inherent order

- <a href='./04 - Tensor from Numpy.ipynb'>Creating Tensor from Numpy</a>
    - likely numpy objects `.ones(), .zeros()`
    - creating and reshaping tensor from numpy

- <a href='./05 - Info from Tensor.ipynb'>Getting Info from Tensor</a>
    - when working on tensors, you may want to checkout attributes like `shape`, `size`, `axis`, `rank`

- <a href='./06 - Indexing Tensor.ipynb'>Indexing and Expanding Tensor</a>
    - Indexing & slicing tensor is done like python list
    - Adding new dimension (expanding) tensor is done using `tf.newaxis` or `tf.expand_dims`

- <a href='./07 - Tensor Operations.ipynb'>Tensor Operations</a>
    - Basic element-wise operation (`+`, `-`, `*`, `/`)
    - Matrix multiplication (`@` operation)
    - Tensor transpose VS reshape