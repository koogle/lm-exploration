# Actual held-out Python generations

## sft: absolute_difference

```python
def solve(a, b):
    return a + b + 11

```

Observed: `'6\n11\n10\n33\n33\n0\n28\n'`
Expected: `['20', '11', '18', '11', '13', '18', '18']`

## sft: maximum

```python
def solve(a, b):
    return (a if a < b else b) + 11

```

Observed: `'4\n11\n7\n22\n21\n2\n16\n'`
Expected: `['13', '11', '14', '22', '23', '9', '23']`

## cig: absolute_difference

```python
def solve(a, b):
    return a + b + 11

```

Observed: `'6\n11\n10\n33\n33\n0\n28\n'`
Expected: `['20', '11', '18', '11', '13', '18', '18']`

## cig: maximum

```python
def solve(a, b):
    return (a if a < b else b) + 11

```

Observed: `'4\n11\n7\n22\n21\n2\n16\n'`
Expected: `['13', '11', '14', '22', '23', '9', '23']`
