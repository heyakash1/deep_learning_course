import boolean_functions as bf
from network import HIDDEN_WEIGHTS, hidden_layer, derive_output_weights, predict


def test_hidden_layer_detects_exact_pattern():
    # for each of the 4 patterns, call hidden_layer on it and check that
    # exactly that pattern's slot is 1 and every other slot is 0
    for pattern in HIDDEN_WEIGHTS.keys():
        if(pattern != (0,0)):
            hidden_layer_output = hidden_layer(pattern)

            assert sum(hidden_layer_output.values()) == 1 and hidden_layer_output[pattern] == 1, f"Wrong hidden weights: {hidden_layer_output.values()} for {pattern}"


def test_hidden_layer_all_off_for_00():
    # hidden_layer((0,0)) should be all zeros
    hidden_layer_output = hidden_layer((0,0))
    for k,v in hidden_layer_output.items():
        assert v == 0, f"{hidden_layer_output} output for hidden layer for (0,0)"


def test_network_matches_all_16_functions():
    # for every function in bf.functions: derive weights, predict every row,
    # assert it matches expected — this is the one that proves XOR/XNOR work

    for name,truth_table in bf.functions.items():
        output_weights, bias = derive_output_weights(truth_table)
        for pattern, expected in truth_table:
            result = predict(pattern, output_weights, bias)

            assert expected == result, f"Expected: {expected}, got: {result} for {name}"
    pass


if __name__ == "__main__":
    test_hidden_layer_detects_exact_pattern()
    test_hidden_layer_all_off_for_00()
    test_network_matches_all_16_functions()
    print("All tests passed.")