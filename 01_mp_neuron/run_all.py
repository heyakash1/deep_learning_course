import mp_neuron as mp
import boolean_functions as bf
def run_all_functions(functions: dict) -> None:

    for name, truth_table, in functions.items():
        n = len(truth_table[0][0])
        result = mp.find_threshold(truth_table,n)
        print(f"{name}: {result}")

if __name__ == "__main__":
    run_all_functions(bf.functions)