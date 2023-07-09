# MicroTorch
MicroTorch is a minimal implementation of PyTorch , NumPy, TensorFlow, and other popular machine learning libraries. It is intended to be used as a learning tool for understanding the how is works neural networks ,autograd, and other machine learning concepts.

## Installation
```bash
pip install microtorch
```

## Usage
```python
import microtorch as mt

# Create a tensor 1 dimension
x = mt.Tensor([1, 2, 3, 4, 5])

# Create a tensor 2 dimension
x = mt.Tensor([[1, 2, 3], [4, 5, 6]])

# Create a tensor 3 dimension
x = mt.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# Create a tensor 4 dimension
x = mt.Tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])

# create a tensor with random values
x = mt.rand(2, 3)

# create a tensor with zeros
x = mt.zeros(2, 3)

# create a tensor with ones
x = mt.ones(2, 3)

# create a tensor with a range
x = mt.arange(0, 10)

# create a tensor with a range and step
x = mt.arange(0, 10, 2)

# create a tensor with a range and step
x = mt.arange(0, 10, 2)

# create a tensor with a range and step
x = mt.arange(0, 10, 2)

# create neural network
model = mt.Sequential(
    mt.Linear(2, 5),
    mt.ReLU(),
    mt.Linear(5, 1),
    mt.Sigmoid()
)

# create loss function
loss_fn = mt.MSELoss()

# create optimizer

optimizer = mt.SGD(model.parameters(), lr=0.01)

# train model

for epoch in range(100):
    # forward
    y_pred = model(x)

    # compute loss
    loss = loss_fn(y_pred, y)

    # zero gradients
    optimizer.zero_grad()

    # backward
    loss.backward()

    # update weights
    optimizer.step()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors
* **[mustafa bozkaya](
    https://github.com/mustafabozkaya)** - *Initial work* - 


