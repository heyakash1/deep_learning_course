import boolean_functions as bf
from network import derive_output_weights, predict


def network_learns_correctly(truth_table: tuple, weights: dict, bias) -> bool:
    """True if predict matches every row of truth_table."""

    for x, expected in truth_table:
        result = predict(x,weights,bias)
        if result != expected:
            return False
    return True


def run_all_functions(functions: dict) -> None:
    correct_count = 0

    for name, truth_table in functions.items():
        weights, bias = derive_output_weights(truth_table)
        correct = network_learns_correctly(truth_table, weights, bias)
        print(name, "correct" if correct else "not correct")
        correct_count += correct

    print(f"{correct_count}/{len(functions)} are correct")


if __name__ == "__main__":
    run_all_functions(bf.functions)