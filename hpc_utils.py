# island_code.py
import pickle
import random
import copy
import argparse
import os

def run_island(pop_file, push_file, gen_num, top_k=5):
    """
    1. Load local population.
    2. Evolve for 1 generation.
    3. Pick top_k individuals and save them in push_file.
    """
    population = load_population(pop_file)  # e.g. a list of (genome, results)
    # Evaluate/training each individual in population
    for ind in population:
        # ind.genome -> train / evaluate -> ind.fitness, etc.
        pass

    # Sort or rank them
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Save top_k to push_file
    top_individuals = population[:top_k]
    with open(push_file, 'wb') as f:
        pickle.dump(top_individuals, f)

    # Save entire updated population for next generation
    next_pop_file = pop_file.replace(f"gen_{gen_num}", f"gen_{gen_num + 1}")
    with open(next_pop_file, 'wb') as f:
        pickle.dump(population, f)


def load_population(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-file", required=True)
    parser.add_argument("--push-file", required=True)
    parser.add_argument("--gen-num", type=int, required=True)
    args = parser.parse_args()

    run_island(args.pop_file, args.push_file, args.gen_num)
