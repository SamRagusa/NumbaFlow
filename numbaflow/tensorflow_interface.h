#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>



typedef struct {
    TF_Session *session;
    TF_Graph *graph;
    TF_Status *status;
    TF_SessionOptions *session_options;
    TF_Output *eval_feeds;
    TF_Output *eval_fetches;
} run_info;


run_info* create_run_info(void);

void close_run_info(run_info* to_close);

float *sess_run(run_info* run_info, uint8_t* compressed_pieces, uint64_t* occupied_bbs, int num_to_eval, int num_piece_bytes);
