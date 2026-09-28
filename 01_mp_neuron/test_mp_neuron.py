import mp_neuron as mp
import boolean_functions as bf

def test_predict_and():
    neuron = mp.MPNeuron(b=2)

    for x, expected in bf.functions["AND"]:
        assert neuron.predict(x) == expected , f"AND failed on input {x}"


def test_threshold_and():
    n = mp.arity(bf.functions["AND"])
    result = mp.find_threshold(bf.functions["AND"],n)

    expected = [2]
    assert result == expected, f"expected {expected}, got {result}"


def test_threshold_or():
    n = mp.arity(bf.functions["OR"])
    result = mp.find_threshold(bf.functions["OR"],n)

    expected = [1]
    assert result == expected , f"expected {expected}, got {result}"

def test_threshold_xor_fails():
    """XOR is not realizable by this neuron — proof of non-monotonicity."""
    n = mp.arity(bf.functions["XOR"])
    result = mp.find_threshold(bf.functions["XOR"],n)

    expected = []
    assert result == expected , f"expected {expected}, got {result}"

def test_threshold_nand_fails():
    """NAND is not realizable either — same reasoning as XOR."""
    n = mp.arity(bf.functions["NAND"])
    result = mp.find_threshold(bf.functions["NAND"],n)

    expected = []
    assert result == expected, f"expected {expected}, got {result}"

def test_threshold_true():
    n = mp.arity(bf.functions["TRUE"])
    result = mp.find_threshold(bf.functions["TRUE"],n)

    expected = [0]
    assert result == expected, f"expected {expected}, got {result}"

def test_threshold_false():
    n = mp.arity(bf.functions["FALSE"])
    result = mp.find_threshold(bf.functions["FALSE"],n)

    expected = [3]
    assert result == expected, f"expected {expected}, got {result}"


if __name__ == "__main__":
    test_predict_and()
    test_threshold_and()
    test_threshold_or()
    test_threshold_xor_fails()
    test_threshold_nand_fails()
    test_threshold_true()
    test_threshold_false()
    print("All tests passed.")