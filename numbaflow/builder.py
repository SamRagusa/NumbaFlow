from cffi import FFI


ffi = FFI()

ffi.set_source(
    module_name='_tensorflow_ffi',
    source='#include "tensorflow_interface.h"',
    sources=["tensorflow_interface.c"],
    libraries=["tensorflow"])

ffi.cdef('''\
typedef ... run_info;

run_info *create_run_info(void);
void close_run_info(run_info*);
float *sess_run(run_info*, uint8_t*, uint64_t*, int, int);
''')



if __name__ == '__main__':
    ffi.compile(verbose=True)