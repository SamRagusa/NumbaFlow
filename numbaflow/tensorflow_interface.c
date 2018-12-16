#include "tensorflow_interface.h"



void numpy_deallocator(void* data, size_t length, void* arg) {
    printf("MAJOR WARNING: this function has not yet been implemented!\n");
}


void free_buffer(void* data, size_t length) {
    free(data);
}


TF_Buffer* read_file(const char* file) {
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}


TF_Graph *get_graph(const char* graphdef_filename) {
    TF_Buffer* graph_def = read_file(graphdef_filename);
    TF_Graph* graph = TF_NewGraph();

    // Import graph_def into graph
    TF_Status* status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
    }

    TF_DeleteStatus(status);
    TF_DeleteBuffer(graph_def);

    return graph;
}


TF_Operation *get_operation(TF_Graph* graph, const char* name) {
    TF_Operation* to_return = TF_GraphOperationByName(graph, name);

    if (to_return == NULL) {
        fprintf(stderr, "ERROR: NULL value returned when getting the following operation from the graph: %s\n", name);
    }

    return to_return;
}


run_info* create_run_info() {
    run_info *holder = (run_info*) malloc(sizeof(run_info));

    TF_SessionOptions* the_sess_options = TF_NewSessionOptions();

    TF_Status* the_status = TF_NewStatus();

    const char* evaluation_filename = "/srv/tmp/combining_graphs_1/TEST_FOR_C_NO_TRT.pb";
    TF_Graph* the_graph = get_graph(evaluation_filename);

    TF_Session* the_session = TF_NewSession(the_graph, the_sess_options, the_status);

    if (TF_GetCode(the_status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to create Session for graph %s\n", TF_Message(the_status));
    }

    TF_Operation* compressed_pieces_placeholder = get_operation(the_graph, "input_parser/piece_filters");
    TF_Operation* occupied_bb_placeholder= get_operation(the_graph, "input_parser/occupied_bbs");
    TF_Operation* eval_logit_tensor = get_operation(the_graph, "value_network/Squeeze");

    TF_Output eval_feeds_to_store[] = {{compressed_pieces_placeholder, 0}, {occupied_bb_placeholder, 0}};
    TF_Output eval_fetches_to_store[] = {{eval_logit_tensor, 0}};

    holder->eval_feeds = malloc(sizeof(eval_feeds_to_store));
    holder->eval_feeds[0] = eval_feeds_to_store[0];
    holder->eval_feeds[1] = eval_feeds_to_store[1];

    holder->eval_fetches = malloc(sizeof(eval_fetches_to_store));
    holder->eval_fetches[0] = eval_fetches_to_store[0];

    holder->session = the_session;
    holder->graph = the_graph;
    holder->status = the_status;
    holder->session_options = the_sess_options;

    return holder;
}


void close_run_info(run_info* to_close) {
    TF_CloseSession(to_close->session, to_close->status);

    if (TF_GetCode(to_close->status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to close TF_Session %s\n", TF_Message(to_close->status));
    }

    TF_DeleteSessionOptions(to_close->session_options);
    TF_DeleteStatus(to_close->status);
    TF_DeleteGraph(to_close->graph);

    free(to_close->eval_feeds);
    free(to_close->eval_fetches);
    free(to_close);
}


float *sess_run(run_info* run_info, uint8_t* compressed_pieces, uint64_t* occupied_bbs, int num_to_eval, int num_piece_bytes) {
    int64_t input_piece_dims[] = {num_piece_bytes};
    int64_t input_occupied_dims[] = {num_to_eval};
    size_t pieces_size = num_piece_bytes * sizeof(uint8_t);
    size_t occupied_size = num_to_eval * sizeof(uint64_t);

    TF_Tensor* input_occupied_tensor = TF_NewTensor(TF_UINT64, input_occupied_dims, 1, occupied_bbs, occupied_size, &numpy_deallocator, 0);
    TF_Tensor* input_pieces_tensor = TF_NewTensor(TF_UINT8, input_piece_dims, 1, compressed_pieces, pieces_size, &numpy_deallocator, 0);

    TF_Tensor* feedValues[] = {input_pieces_tensor, input_occupied_tensor};

    size_t output_size = num_to_eval * sizeof(float);
    TF_Tensor* output_tensor =  TF_AllocateTensor(TF_FLOAT, input_occupied_dims, 1, output_size);                //Triple check this won't cause any memory leaks (or things of that sort)

    TF_Tensor* fetchValues[] = {output_tensor};

    TF_SessionRun(
        run_info->session,
        NULL, // run_options
        run_info->eval_feeds,
        feedValues,
        2,    // The number of input tensors,
        run_info->eval_fetches,
        fetchValues,
        1,    // The number of output tensors
        NULL, // target_opers
        0,
        NULL, // run_metadata
        run_info->status);

    if (TF_GetCode(run_info->status) != TF_OK) {
        fprintf(stderr, "ERROR: Not okay TF_Status after TF_SessionRun call %s\n", TF_Message(run_info->status));
    }

    return (float*) TF_TensorData(output_tensor);            // This (or something related to it) may be causing the return values to not be the computed values
}

