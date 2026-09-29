# 02 – Perceptron

## What this is
Frank Rosenblatt, an American psychologist, proposed the classical
perceptron model in 1958. It is a more general computational model than
the McCulloch–Pitts neuron.

The main differences: the perceptron has numerical weights for its inputs
instead of fixed ones, and a mechanism for learning those weights (and the
threshold) from data. Inputs are also no longer restricted to boolean
values in general, though this project still uses boolean inputs to
compare directly against the MP neuron.

## Files
- `boolean_functions.py`: truth tables for all 16 two-input boolean functions, stored as a dictionary.
- `perceptron.py`: the `Perceptron` class (`predict` and `fit`, which learns the weights and bias) and `arity`.
- `run_all.py`: trains a fresh perceptron on each of the 16 functions and prints the result.
- `test_perceptron.py`: tests for prediction, seed reproducibility, and convergence.

## How to run
From inside the `02_perceptron` folder (the files import each other by name):

```
pip install numpy
python run_all.py
python test_perceptron.py
```

## Results
| Function    | Converged | Epochs |
|-------------|-----------|--------|
| FALSE       | yes       | 2      |
| AND         | yes       | 6      |
| A_AND_NOT_B | yes       | 3      |
| A           | yes       | 2      |
| NOT_A_AND_B | yes       | 3      |
| B           | yes       | 2      |
| XOR         | no        | 100    |
| OR          | yes       | 3      |
| NOR         | yes       | 5      |
| XNOR        | no        | 100    |
| NOT_B       | yes       | 3      |
| B_IMPLIES_A | yes       | 5      |
| NOT_A       | yes       | 4      |
| A_IMPLIES_B | yes       | 4      |
| NAND        | yes       | 7      |
| TRUE        | yes       | 1      |

14/16 functions converged.

## Comparison with the MP neuron
The MP neuron could only realize 4 of the 16 functions: FALSE, AND, OR and
TRUE. Two problems held it back. First, its weights were always 1, so it
could only get more "yes" as it saw more 1s — it could never learn to say
"no" when an input turned on. Second, every input counted the same way, so
it couldn't tell `(1, 0)` from `(0, 1)`.

The perceptron fixes both. Its weights are learned, and they can go
negative, so it can fire less as an input turns on — this is what makes
NOT_A, NAND, NOR and the two implications work. And each input gets its own
weight, so `(1, 0)` and `(0, 1)` can lead to different outputs — this is
what makes A, B, A_AND_NOT_B and NOT_A_AND_B work. That's why the perceptron
realizes 14 of 16 instead of 4.

## Why XOR and XNOR still fail
A perceptron draws one straight line through the input space and separates
the 1s from the 0s on either side of it. XOR's 1s and 0s can't be separated
by any single line — you'd need two lines, crossing each other, to fence
them apart.

This is easy to see as a contradiction. For XOR, the perceptron would need,
using `w0` for the bias and `w1`, `w2` for the two input weights:

- `(0,0) → 0`, so `w0 < 0`
- `(0,1) → 1`, so `w0 + w2 >= 0`
- `(1,0) → 1`, so `w0 + w1 >= 0`
- `(1,1) → 0`, so `w0 + w1 + w2 < 0`

Add the middle two together: `2*w0 + w1 + w2 >= 0`. But the last line says
`w1 + w2 < -w0`, so `2*w0 + w1 + w2 < w0`, and `w0` is negative. That means
`2*w0 + w1 + w2` would have to be both `>= 0` and less than a negative
number, which is impossible. So no weights exist that satisfy all four
rows at once — that's why training never converges, and why `run_all.py`
reports 100 epochs for XOR. XNOR gives the same contradiction with the
inequalities flipped.

## Notes
Initial weights are drawn randomly in [-1, 1). Passing `seed` to `Perceptron`
makes a run reproducible; `run_all.py` uses a fixed seed so the table above
can be reproduced exactly. The tests instead try several different seeds
per function and assert only on whether it converges, not on the exact
epoch count or weights, since those change with the seed while the
converged/not-converged result does not.

`run_all.py` also re-checks every trained perceptron against the truth
table independently of `fit`'s own convergence flag, and flags any
disagreement between the two. They agree on all 16 functions.