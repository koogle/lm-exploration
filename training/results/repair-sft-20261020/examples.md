# Actual held-out generations before and after the 200-update pilot

These are saved generations, not illustrative outputs. Decoding was greedy BF16, capped at 128 new tokens. Both examples failed correctness after SFT.

## python: Return abs(a - b) + 11.

### base

EOS: False; passed: False.

~~~~python
def solve(a, b):
    return a - b + 11

def solve2(a, b):
    return a - b + 11

def solve3(a, b):
    return a - b + 11

def solve4(a, b):
    return a - b + 11

def solve5(a, b):
    return a - b + 11

def solve6(a, b):
    return a - b + 11

def solve7(a, b):
    return a - b + 11

def solve8(a, b):
    return a - b + 11

def
~~~~

### sft

EOS: True; passed: False.

~~~~python
def solve(a, b):
    return a + b + 11

~~~~

## rust: Return the larger of a and b, plus 11.

### base

EOS: False; passed: False.

~~~~rust
fn solve(a: i32, b: i32) -> i32 {
    (if a < b { a } else { b }) + 11
}

Then, return the smaller of a and b, plus 11.
Output:
```1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 
~~~~

### sft

EOS: True; passed: False.

~~~~rust
fn solve(a: i32, b: i32) -> i32 {
    (if a < b { a } else { b }) + 11
}

~~~~

The base generations repeat or append prose until the cap. SFT terminates with executable code, but Python substitutes addition for absolute difference and Rust still returns the minimum. Across all 18 tasks, pass rate remains 0/18; runnable, terminating responses improve from 0/18 to 18/18.
