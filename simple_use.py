import numbaflow as nf
import numpy as np


"""
Before running you must build the cffi library, which can be done by running the builder.py file (which is within
the numbaflow package). 
"""


if __name__ == "__main__":
    eval_fn, closer_fn = nf.create_eval_fn()

    dummy_pieces = np.array([96], dtype=np.uint8)
    dummy_occupied = np.ones(1, np.uint64)

    results = eval_fn(dummy_pieces, dummy_occupied)

    print("Results:", results)

    closer_fn()
