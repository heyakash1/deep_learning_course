import boolean_functions as bf
import numpy as np
from perceptron import Perceptron, arity
from run_all import learns_correctly


def test_predict_and():
    weights = np.array([-2.0,1.0,1.0])
    p = Perceptron(2)
    p.w = weights
    for row, expected in bf.functions["AND"]:
        assert p.predict(row) == expected , f"AND failed on input {row}"

def test_seed_is_reproducible():
    p1 = Perceptron(2,seed=2)
    p2 = Perceptron(2,seed=2)

    assert np.array_equal(p1.w,p2.w), f"SEED is not reproducible"

def test_separable_functions_converge():
    that_converge = {k:v for k,v in bf.functions.items() if k not in ("XOR","XNOR")}
    for name, truth_table in that_converge.items():
        p = Perceptron(arity(truth_table))

        converged, epochs = p.fit(truth_table)
        assert converged, f"{name} did not converge"


def test_non_separable_functions_do_not_converge():
    that_dont_converge = {key:bf.functions[key] for key in ("XOR","XNOR") if key in bf.functions}
    for name, truth_table in that_dont_converge.items():
        p = Perceptron(arity(truth_table))

        converged, epochs = p.fit(truth_table)
        assert not converged, f"{name} converged"

if __name__ == "__main__":
    test_predict_and()
    test_seed_is_reproducible()
    test_separable_functions_converge()
    test_non_separable_functions_do_not_converge()
    print("All tests passed.")