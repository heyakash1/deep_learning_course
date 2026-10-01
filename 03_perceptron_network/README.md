# 03 – Network of Perceptrons

## What this is
A single perceptron draws one straight line to separate its 1s from its 0s.
That's why XOR failed in the last topic — its two "1" points and two "0"
points can't be split apart by any one line.

This topic sidesteps that problem instead of solving it. The network has a
hidden layer of 4 units, one for each possible 2-input pattern — `(0,0)`,
`(0,1)`, `(1,0)`, `(1,1)`. Each hidden unit is wired so it fires only for
its own exact pattern and stays silent for every other input. So instead of
asking "can a line separate these points?", the network just asks "which
one of the 4 exact inputs is this?" — and that question has no geometry
problem at all, so it works for every boolean function, separable or not.

## Files
- `boolean_functions.py`: truth tables for all 16 two-input boolean functions (reused from 02).
- `network.py`: `HIDDEN_WEIGHTS` and `HIDDEN_BIAS` (the fixed hidden layer), `hidden_layer` (runs all 4 detectors on an input), `derive_output_weights` (works out the output layer's weights and bias from a truth table), and `predict` (combines both layers into a final 0/1 answer).
- `run_all.py`: builds a network for each of the 16 functions and checks it against the full truth table.
- `test_network.py`: checks that each hidden unit fires only for its own pattern, that `(0,0)` turns every hidden unit off, and that the network gets all 16 functions right.

## How to run
From inside the `03_perceptron_network` folder (the files import each other by name):

```
python run_all.py
python test_network.py
```

## The hidden layer
Each hidden unit is a threshold unit with a fixed bias of -2:

| Pattern | Weights (w1, w2) |
|---------|-------------------|
| (0,0)   | (0, 0) — doesn't matter, see below |
| (0,1)   | (-1, 2) |
| (1,0)   | (2, -1) |
| (1,1)   | (1, 1)  |

A unit fires when `w1*x1 + w2*x2 - 2 >= 0`. For `(0,1)`, `(1,0)` and
`(1,1)`, the weights are chosen so this is only ever true at that unit's
own pattern — any other input brings the sum below 0.

`(0,0)` is special: since both inputs are 0, any weight multiplied by 0 is
0, so the sum is always `0 + 0 - 2 = -2` no matter what the weights are.
That unit can never fire — not a bug, just a fact of the fixed bias. But
this turns out to be fine: `(0,0)` is the only input where *every* hidden
unit is off at once, so "all zeros" still uniquely identifies it, the same
way a single lit-up unit identifies the other three inputs.

## Deriving the output layer
Once the hidden layer tells you exactly which pattern the input was, the
output layer just has to look at the truth table and decide: does this
pattern's row say 1 or 0?

- If a pattern's expected output is 1, its output weight is set to `-bias`,
  so when that unit fires, the sum lands exactly on 0 and the network
  outputs 1.
- If a pattern's expected output is 0, its output weight is set to
  `-bias - 1`, landing one below the threshold, so the network outputs 0.
- The output bias itself is set from the `(0,0)` row alone: 1 if `(0,0)`'s
  expected output is 1, else -1. This works because `(0,0)` turns every
  hidden unit off, so the bias is the only thing deciding that case.

Worked example, XOR (bias = -1, since `(0,0) → 0`):
- `(0,1) → 1`, so weight = `-bias = 1`
- `(1,0) → 1`, so weight = `-bias = 1`
- `(1,1) → 0`, so weight = `-bias - 1 = 0`

Checking all four: `(0,0)` → bias alone = -1, no fire, correct. `(0,1)` →
`1 - 1 = 0`, fires, correct. `(1,0)` → `1 - 1 = 0`, fires, correct. `(1,1)`
→ `0 - 1 = -1`, no fire, correct. XOR works.

## Results
16/16 functions correct, including XOR and XNOR, which the single
perceptron in topic 02 could not do.

## Comparison across all three topics
| Topic         | Realizes |
|---------------|----------|
| 01 MP neuron  | 4 / 16   |
| 02 Perceptron | 14 / 16  |
| 03 Network    | 16 / 16  |

The MP neuron could only grow more "yes" as more inputs turned on, and
couldn't tell `(1,0)` from `(0,1)` apart. The perceptron fixed both of
those with learned, signed weights, but still hit a wall on XOR and XNOR,
since no single line separates their points. The network gets past that
wall by not drawing a line at all — it identifies the exact input first,
then just looks up the answer.