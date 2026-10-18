# Qualitative review and counterexamples

This is an explicit manual sample review, not an automated quality benchmark.
Review covered the first 240 characters of all 90 first-seed completions and full
second-seed completions for the baseline and three low-repetition presets
(T=.85/p=.95/penalty1.1, T=1/p=.95/penalty1.1, T=.85/p=.90/penalty1.15).
The examples below are selected to illustrate tradeoffs. All 170 unfiltered
completions remain available separately. No generated code was executed.

The review did not identify a preset that consistently provided a readable,
relevant continuation. Settings with fewer exact repeated n-grams still mixed
unrelated topics, contradicted their own statements, invented words or names,
and produced malformed or incorrect Python. These problems matter even when
factual accuracy is not the priority.

## Breaking a loop does not establish coherence

Prompt: `The solar system consists of`, seed 1000. The greedy completion repeats
“a solar system” throughout. T=.85/p=.95/penalty1.1 instead starts:

> approximately 13,000 cameras, which transmit information from one location to another through a variety of mediums and infrared radiation sources.

It then drifts to a power station and Greek absorption, rather than explaining
the solar system. This is a successful reduction in exact repetition, not a
successful answer. In the second seed, the same preset continues with heat
transfer and quenching rather than a stable explanation.

## A matched penalty change can spoil a useful initial answer

Prompt: `2 + 2 =`, seed 1008, T=1/top-p=.95 held fixed:

Penalty 1.0:

````text
 4
Sum of all the values of x in the set (2, 2, 2, 2, 2, 4, 2, 4, 4) is
(2 + 2 + 2 + 2 + 2 + 2 + 2)
x1
x2
x3
x4
0
(c) (c) The range of the equation x2 + 5x + 5 = 4 is (2, 0)
(2, 4)
2 2 4 0
Some letter
````

Penalty 1.1:

````text
 3 Solving for x
2 + 2 = 3 x = 3 + 2 y = 6 x = 3 y = 6-2 t = 5

The larger √2 > √3 ≤ x < 5 is a positive solution that covers −∞ √2 = 0. (If 0 > 0, then 0 > −∞)
The larger √2 > √3 is 5.
The larger 4 ≤
````

The baseline starts with the correct answer, 4; penalty1.1 starts with 3.
Both drift afterward. With seed 2008 the direction reverses: the baseline
starts with 2 and penalty1.1 starts with 4. This is a tradeoff example, not
evidence of a statistically established drop or improvement in arithmetic accuracy.

## Stronger penalties can damage code

Prompt `def add(a, b):` with four spaces after its newline, seed 1004.
Temperature .85 and top-p .90 held fixed:

Penalty 1.05, first generated function body:

````python
def add(a, b):
     a += b
     return a
````

Penalty 1.15, first generated function body:

````python
def add(a, b):
     a += 1
     b += 1
     return a + b
````

The weaker-penalty body implements addition for ordinary numbers. The
stronger-penalty body increments both arguments before adding them. Both
completions continue beyond the function, so this is a prefix comparison,
not a full-program pass/fail score. The stronger penalty has lower aggregate
repetition but does not preserve this useful function body.

Greedy `read_config` also starts with the useful `with open(...): json.load(f)`
pattern before generating repeated variants of the function. A task-specific
stop rule might preserve such useful prefixes, but that rule was not tested
here and should not be presented as a measured fix.

## Preset decision

T=.85/top-p=.95/penalty1.1 is worth keeping as an **anti-loop exploration
candidate**, alongside the unpenalized T=1/top-p=.95 control. It is not established
as a quality improvement. The minimum-repetition setting, T=1/penalty1.1, often
produces especially disordered prose or code. Do not choose it just because
its repetition metric is smallest. No reliable code preset or general default
was identified. Preserve the negative result and investigate model/data quality
with the independent baseline study before claiming the generation is fixed.
