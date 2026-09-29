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
    pass

def test_non_separable_functions_do_not_converge():
    pass

if __name__ == "__main__":
    test_predict_and()
    test_seed_is_reproducible()
    test_separable_functions_converge()
    test_non_separable_functions_do_not_converge()
    print("All tests passed.")