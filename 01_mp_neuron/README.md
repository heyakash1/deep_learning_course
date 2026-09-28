# 01 – McCulloch-Pitts Neuron

## What this is
The McCulloch-Pitts (MP) neuron is the earliest simplified model of a biological
neuron, proposed by Warren McCulloch and Walter Pitts in 1943. It is the
foundation that artificial neural networks and modern deep learning build on.

This version takes 0/1 inputs, adds them up, and outputs 1 if the sum reaches a
threshold `b`, otherwise 0. All weights are fixed at 1 and there are no
inhibitory inputs. The only thing to choose is `b`, so this project searches for
the `b` that reproduces each boolean functio


## Files
- `boolean_functions.py`: truth tables for all 16 two-input boolean functions, stored as a dictionary.
- `mp_neuron.py`: the `MPNeuron` class (threshold rule), `find_threshold` (tries each `b` against a truth table) and `arity`.
- `run_all.py`: runs the threshold search on all 16 functions and prints the result for each.
- `test_mp_neuron.py`: tests for the neuron and the threshold search.


## How to run
From inside the `01_mp_neuron` folder (the files import each other by name):

```
python run_all.py
python test_mp_neuron.py
```
No dependencies beyond standard Python.


## Results
| Function    | Threshold found |
|-------------|-----------------|
| FALSE       | 3               |
| AND         | 2               |
| A_AND_NOT_B | none            |
| A           | none            |
| NOT_A_AND_B | none            |
| B           | none            |
| XOR         | none            |
| OR          | 1               |
| NOR         | none            |
| XNOR        | none            |
| NOT_B       | none            |
| B_IMPLIES_A | none            |
| NOT_A       | none            |
| A_IMPLIES_B | none            |
| NAND        | none            |
| TRUE        | 0               |


## Why most functions have no threshold
This neuron does one simple thing: it counts how many inputs are 1 and fires
(outputs 1) if that count reaches the threshold `b`. Two limits follow from that.

1. **More 1s can never turn the output off.** Flipping an input from 0 to 1 can
   only raise the count, so a neuron that already fires keeps firing. Functions
   like NOT_A, NAND, NOR and XOR need the output to drop when an input goes
   from 0 to 1, so no threshold can make them work.

2. **It only sees how many inputs are 1, not which ones.** `(1, 0)` and `(0, 1)`
   both have a count of 1, so the neuron must give them the same output. Any
   function that treats the two inputs differently (A, B, A_AND_NOT_B and the
   implications) is impossible for the same reason. A and B never decrease, but
   they still fail on this rule.

Only AND (`b = 2`), OR (`b = 1`), TRUE (`b = 0`) and FALSE (`b = 3`) pass both
tests.

This is a limit of *this simplified neuron* (all weights fixed at 1, no
inhibitory inputs), not of boolean functions in general. 14 of the 16 functions
are linearly separable. XOR and XNOR are the only two that are not. A neuron
with adjustable weights (the perceptron, next topic) can realize all 14
separable functions.


## Notes
The threshold search tries `b` from `0` to `n + 1` (with `n = 2` that is 0, 1,
2, 3). The extra value at the top matters for FALSE: the largest possible sum is
`n`, so only a threshold of `n + 1` makes the neuron never fire. An earlier
version searched only `0..n`, which made FALSE come out as "none". TRUE (`b = 0`,
always fires) and FALSE (`b = 3`, never fires) are the two constant functions,
and both are degenerate neurons that ignore their inputs.