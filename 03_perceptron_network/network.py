import boolean_functions as bf


HIDDEN_WEIGHTS = {
    (0,0) : (0,0),
    (0,1) : (-1,2),
    (1,0) : (2,-1),
    (1,1) : (1,1),
}
HIDDEN_BIAS = -2


def _fires(w: tuple, x: tuple, bias: int) -> int:
    aggregation = sum(a * b for a, b in zip(w,x))
    aggregation += bias
    if aggregation >= 0:
        return 1
    return 0


def hidden_layer(x: tuple) -> dict:
    output = {}
    for pattern, weights in HIDDEN_WEIGHTS.items():
        output[pattern] = _fires(weights, x, HIDDEN_BIAS)

    return output


def derive_output_weights(truth_table: tuple) -> tuple:
    tt = dict(truth_table)
    weights = {}
    bias = 1 if tt[(0,0)]== 1 else -1
    for pattern, expected_output in tt.items():
        weights[pattern] = ((-bias)-1) if expected_output == 0 else -bias

    return (weights,bias)


def predict(x: tuple, weights: dict, bias) -> int:
    output = hidden_layer(x)
    aggregate = 0
    for pattern, hidden_output in output.items():
        aggregate += weights[pattern]*hidden_output

    aggregate += bias
    if aggregate >= 0:
        return 1
    return 0


if __name__ == "__main__":
    weights, bias = derive_output_weights(bf.functions["XNOR"])
    for x, expected in bf.functions["XNOR"]:
        result = predict(x, weights, bias)
        print(x, "expected:", expected, "got:", result)