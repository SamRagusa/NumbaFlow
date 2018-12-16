# NumbaFlow
NumbaFlow uses the TensorFlow [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h) to perform low latency and high-throughput inference from within Numba JIT compiled functions.    

## Dependencies
- [TensorFlow for C](https://www.tensorflow.org/install/lang_c)
- [Numba](https://github.com/numba/numba)
- [NumPy](https://github.com/numpy/numpy)

## Notes
- This project is still in development and thus as of now it does not currently work
- Currently it's being built specifically to score chess boards/moves for the [Batch First](https://github.com/SamRagusa/Batch-First) chess engine, but once that is completed, generalizing the functionality will become the priority  

## License
NumbaFlow is licensed under the MIT License.  The full text can be found in the LICENSE.txt file.