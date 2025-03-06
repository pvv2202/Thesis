import pickle
import glob
import heapq
import os

def run(subpop_dirs, gen_num, global_heap_file="global_heap.pkl", top_k=5):
    """
    1. Load the master min-heap (or max-heap) from global_heap_file.
    2. Read each subpop_i's push_gen_X.pkl
    3. Insert those individuals into the heap
    4. Extract the top individuals (by fitness)
    5. Write them back out to each subpop_i as pull_gen_X.pkl
    6. Save the updated heap to disk
    """
    # 1. Load existing global heap
    global_heap = []
    if os.path.exists(global_heap_file):
        with open(global_heap_file, 'rb') as f:
            global_heap = pickle.load(f)

    # 2. Collect new individuals from each subpop
    new_inds = []
    for subdir in subpop_dirs:
        push_file = os.path.join(subdir, f"push_gen_{gen_num}.pkl")
        if not os.path.exists(push_file):
            print(f"Warning: no push file for {subdir} gen {gen_num}")
            continue
        with open(push_file, 'rb') as f:
            top_island_inds = pickle.load(f)
        new_inds.extend(top_island_inds)

    # 3. Insert them into the global heap
    #    Assume we want a "max-heap" by fitness, so we might store negative fitness for heapq, or do a custom comparator.
    for ind in new_inds:
        # Suppose ind.fitness is higher-is-better
        # For Python's `heapq` (which is a min-heap), we do negative fitness
        entry = (-ind.fitness, ind)  # store (negative_fitness, actual_individual)
        heapq.heappush(global_heap, entry)

    # 4. Possibly shrink the heap if it’s too large
    #    e.g. keep only top 100 or top 1000
    while len(global_heap) > 1000:
        heapq.heappop(global_heap)

    # 5. Extract top_k from the heap to send back to each subpop
    best_global = []
    temp = []
    for _ in range(top_k):
        if len(global_heap) == 0:
            break
        best_global.append(heapq.heappop(global_heap))

    # Put them back (so the heap is unchanged except for new additions)
    for entry in best_global:
        heapq.heappush(global_heap, entry)

    # 6. Write these best_global individuals to each subpop’s pull_gen_X.pkl
    for subdir in subpop_dirs:
        pull_file = os.path.join(subdir, f"pull_gen_{gen_num}.pkl")
        with open(pull_file, 'wb') as f:
            # Re-extract actual individuals from (neg_fitness, ind)
            pulled_inds = [entry[1] for entry in best_global]
            pickle.dump(pulled_inds, f)

    # 7. Save the updated heap
    with open(global_heap_file, 'wb') as f:
        pickle.dump(global_heap, f)


if __name__ == "__main__":
    # For real usage, parse arguments or environment variables
    subpop_dirs = ["subpop_0", "subpop_1", ...]
    gen_num = 0
    run(subpop_dirs, gen_num)
