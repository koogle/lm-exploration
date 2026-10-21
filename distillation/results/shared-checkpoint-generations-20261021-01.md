# Generations across shared-tokenizer checkpoints

Actual native-model generations on one L4, FP32, greedy decoding, 128 new tokens maximum. Same prompt IDs and end-of-text prefix for every checkpoint. No chat template, sampling, repetition penalty or output cleanup. The 0M checkpoint is after tokenizer conversion, not the original 124M model. Outputs below include the complete decoded continuation, including special markers and repetition. Raw token IDs and stop reasons are in the accompanying JSON.

[Modal run](https://modal.com/apps/radiantai/main/ap-HVJ6KvuIfRKJXQpBiBL03l) · [Raw JSON](shared-checkpoint-generations-20261021-01.json)

## add

Prompt:

````python
def add(a, b):
    """Return the sum of a and b."""
````

### 0 training tokens (step 0)

Stop: `length`; 128 generated IDs.

````python
   n =
   n =
    return =
   n =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set()
   set =set
````

### 1,000,000 training tokens (step 147)

Stop: `length`; 128 generated IDs.

````python
    return a + b

def get(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all(a, b):
    """Return the sum of all elements in a and all elements in b."""
    return a + b

def get_all_all(a, b):
    """Return the sum of all elements in a and all elements in b."""
    return a + b

def get_all_all_all(a, b):
    """Return the sum of all elements in a and all elements in b."""
    return a + b

def get_all_all_all_all(a, b):

````

### 5,000,000 training tokens (step 715)

Stop: `length`; 128 generated IDs.

````python
    return a + b

def get(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all_all_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all_all_all_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get
````

### 10,000,000 training tokens (step 1439)

Stop: `length`; 128 generated IDs.

````python
    return a + b
def get(a, b):
    return a + b
def getAll(a, b):
    return a + b
def getAllAll(a, b):
    return a + b
def getAllAllAll(a, b):
    return a + b
def getAllAllAllAll(a, b):
    return a + b
def getAllAllAllAll(a, b):
    return a + b
def getAllAllAllAllAll(a, b):
    return a + b
def getAllAllAllAllAll(a, b):
    return a + b
def get
````

### 25,000,000 training tokens (step 3588)

Stop: `length`; 128 generated IDs.

````python
    return a + b

def get(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all_all_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get_all_all_all_all(a, b):
    """Return the sum of a and b."""
    return a + b

def get
````

### 50,000,000 training tokens (step 7162)

Stop: `length`; 128 generated IDs.

````python
    return a + b

def main():
    print "Adding 1 to 2"
    print "Adding 2 to 3"
    print "Adding 3 to 4"
    print "Adding 4 to 5"
    print "Adding 5 to 6"
    print "Adding 6 to 7"
    print "Adding 7 to 8"
    print "Adding 8 to 9"
    print "Adding 9 to 10"
    print "Adding 10 to 11"
    print "Adding 11 to 12"
    print "Adding
````

## unique

Prompt:

````python
def unique(items):
    """Return a list with duplicates removed, preserving order."""
````

### 0 training tokens (step 0)

Stop: `length`; 128 generated IDs.

````python
    =
    =
    =
    =
   def
   def<|vision_pad|><|vision_pad|><|vision_pad|><|vision_pad|><|vision_pad|><|vision_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|>
````

### 1,000,000 training tokens (step 147)

Stop: `length`; 128 generated IDs.

````python
    return [item for item in items if not item.is_unique]

def uniqueBy(items):
    """Return a list with duplicates removed, pre order."""
    return [item for item in items if not item.is_unique]

def uniqueByBy(items):
    """Return a list with duplicates removed, pre order."""
    return [item for item in items if not item.is_unique]

def uniqueByByBy(items):
    """Return a list with duplicates removed, pre order."""
    return [item for item in items if not item.is_unique]

def uniqueByByBy(items):
    """Return a list with duplicates removed, pre order
````

### 5,000,000 training tokens (step 715)

Stop: `length`; 128 generated IDs.

````python
    return [item for item in items if not item in items]

def unique_items(items):
    """Return a list with duplicates removed, keeping order."""
    return [item for item in items if not item in items]


def unique_items_items(items):
    """Return a list with duplicates removed, keeping order."""
    return [item for item in items if not item in items]


def unique_items_items(items):
    """Return a list with duplicates removed, keeping order."""
    return [item for item in items if not item in items]


def unique_items_items_items(items):
    """Return a list with duplicates removed, keeping order."""

````

### 10,000,000 training tokens (step 1439)

Stop: `length`; 128 generated IDs.

````python
    return [item for item in items if item not in items]

def uniqueByIndex(index):
    """Return a list with duplicates removed, preserving order."""
    return [item for item in index if item not in index]

def uniqueByIndexByIndex(index):
    """Return a list with duplicates removed, preserving order."""
    return [item for item in index if item not in index]

def uniqueByIndexByIndexByIndex(index):
    """Return a list with duplicates removed, preserving order."""
    return [item for item in index if item not in index]

def uniqueByIndexByIndexByIndexByIndex(index):

````

### 25,000,000 training tokens (step 3588)

Stop: `length`; 128 generated IDs.

````python
    return [item for item in items if item not in items]

def find_by_name(name):
    """Returns a list with duplicates removed, preventing order."""
    return [item for item in sorted(list(items)) if item not in items]

def find_by_name_and_index(name, index):
    """Returns a list with duplicates removed, preventing order."""
    return [item for item in sorted(list(items)) if item not in items]

def find_by_name_and_index_and_index(name, index, index):
    """Returns a list with duplicates removed, prevent order."""
    return [item for item in sorted(list(items)) if item not
````

### 50,000,000 training tokens (step 7162)

Stop: `eos`; 40 generated IDs.

````python
    return [item for item in items if item not in unique]

def unique_items():
    """Return a list with duplicates removed, preserving order."""
    return [item for item in unique]
<|endoftext|>
````

## factorial

Prompt:

````python
def factorial(n):
    """Return n factorial for a non-negative integer n."""
````

### 0 training tokens (step 0)

Stop: `length`; 128 generated IDs.

````python
    =
   n =
   n =
    return =
   n =1
   if =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   ifdef =1
   if
````

### 1,000,000 training tokens (step 147)

Stop: `length`; 128 generated IDs.

````python
    return n * n - 1

def factorial(n):
    """Return n factorial for a non-negative integer n."""
    return n * n - 1

def factorial2(n):
    """Return n factorial for a non-negative integer n."""
    return n * n - 1

def factorial3(n):
    """Return n factorial for a non-negative integer n."""
    return n * n - 1

def factorial4(n):
    """Return n factorial for a non-negative integer n."""
    return n * n - 1

def factorial5(n):
    """
````

### 5,000,000 training tokens (step 715)

Stop: `eos`; 49 generated IDs.

````python
    return n + 1

def main():
    """Main function."""
    n = 1
    for i in range(1, n):
        print(i)
    print(n)
    print('Done')
    return 0<|endoftext|>
````

### 10,000,000 training tokens (step 1439)

Stop: `length`; 128 generated IDs.

````python
    return n * Math.log(n)

def log(n):
    """Return n log n."""
    return log(n)

def log2(n):
    """Return n log n."""
    return log(n)

def log3(n):
    """Return n log n."""
    return log(n)

def log4(n):
    """Return n log n."""
    return log(n)

def log5(n):
    """Return n log n."""
    return log(n)

def log6(n):
    """Return n log n."""
    return log(n)

def log7(n):
    """Return n log n."""
    return log(n)

def log
````

### 25,000,000 training tokens (step 3588)

Stop: `eos`; 36 generated IDs.

````python
    return n % 2

def main():
    n = 1
    while n < 1:
        n = n * 2
    print(n)<|endoftext|>
````

### 50,000,000 training tokens (step 7162)

Stop: `length`; 128 generated IDs.

````python
    return n * n

def main():
    n = 100
    n = 1000
    n = 10000
    n = 100000
    n = 1000000
    n = 10000000
    n = 100000000
    n = 1000000000
    n = 10000000000
    n = 1000000
````
