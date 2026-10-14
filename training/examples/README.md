# Minimal NumPy training example

[`numpy_xor.py`](numpy_xor.py) trains a `Linear(2, 5) → ReLU → Linear(5, 1)`
network on 20 fixed binary examples. The second input is the complement of the
first, except every third entry matches it; targets are their XOR.

The example includes handwritten backpropagation, bias folded into an extra
weight column, squared loss and gradient descent. It uses seed `0`, four batches
of five per epoch, a learning rate of `0.01` and 1,000 epochs. Loss gradients are
divided by the batch output size to optimize mean squared loss.

Run from `training/` with Python and NumPy available:

```sh
python examples/numpy_xor.py
```

With the repository's uv environment already installed, use
`uv run python examples/numpy_xor.py` instead.

Measured training loss with Python 3.12.14 and NumPy 2.3.5:

```text
Epoch    0: loss = 0.57908271
Epoch  100: loss = 0.03758829
Epoch  200: loss = 0.01351551
Epoch  300: loss = 0.00413844
Epoch  400: loss = 0.00113843
Epoch  500: loss = 0.00029152
Epoch  600: loss = 0.00007271
Epoch  700: loss = 0.00001784
Epoch  800: loss = 0.00000437
Epoch  900: loss = 0.00000105
Epoch 1000: loss = 0.00000026
```

These losses are evaluated on the same 20 training examples.
