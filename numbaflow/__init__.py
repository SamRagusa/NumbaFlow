import cffi

ffi = cffi.FFI()

import numpy as np

from numba import cffi_support

import numbaflow._tensorflow_ffi as _tensorflow_ffi



cffi_support.register_module(_tensorflow_ffi)

get_info_struct = _tensorflow_ffi.lib.create_run_info
close_info_struct = _tensorflow_ffi.lib.close_run_info
sess_run = _tensorflow_ffi.lib.sess_run


def create_eval_fn():
    info_struct = get_info_struct()

    def evaluator(pieces, masks):
        returned_pointer = sess_run(
            info_struct,
            ffi.cast("uint8_t *", ffi.from_buffer(pieces)),
            ffi.cast("uint64_t *", ffi.from_buffer(masks)),
            len(masks),
            len(pieces))

        result_buff = ffi.buffer(returned_pointer)
        return np.frombuffer(result_buff, dtype=np.float32)

    closer = lambda : close_info_struct(info_struct)

    return evaluator, closer
