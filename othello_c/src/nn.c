#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime_c_api.h>

/*
 * One NN instance owns one ORT session. The model's input is named
 * dynamically by the exporter; we look up names via the session metadata
 * rather than hardcoding.
 *
 * Output 0 is the policy log-softmax (B, ACTION_SIZE); we exp() it on the C
 * side, matching OthelloNNet.predict_batch in /othello/pytorch/NNet.py:96.
 * Output 1 is the value (B, 1); we squeeze to (B,).
 */

struct NN {
    const OrtApi*       api;
    OrtEnv*             env;
    OrtSessionOptions*  opts;
    OrtSession*         session;
    OrtMemoryInfo*      mem_info;
    OrtAllocator*       alloc;
    char*               input_name;
    char*               policy_name;
    char*               value_name;
};

#define ORT_CHECK(call) do { \
    OrtStatus* _st = (call); \
    if (_st != NULL) { \
        const char* _msg = nn->api->GetErrorMessage(_st); \
        fprintf(stderr, "ORT error: %s\n", _msg ? _msg : "(null)"); \
        nn->api->ReleaseStatus(_st); \
        return 1; \
    } \
} while (0)

int nn_load(NN** out, const char* onnx_path, int use_cuda) {
    NN* nn = calloc(1, sizeof(NN));
    if (!nn) return 1;
    nn->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!nn->api) {
        fprintf(stderr, "OrtGetApiBase: incompatible API version %d\n", ORT_API_VERSION);
        free(nn);
        return 1;
    }

    ORT_CHECK(nn->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "othello_c", &nn->env));
    ORT_CHECK(nn->api->CreateSessionOptions(&nn->opts));
    ORT_CHECK(nn->api->SetIntraOpNumThreads(nn->opts, 1));
    ORT_CHECK(nn->api->SetInterOpNumThreads(nn->opts, 1));
    ORT_CHECK(nn->api->SetSessionGraphOptimizationLevel(nn->opts, ORT_ENABLE_ALL));

    if (use_cuda) {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id = 0;
        OrtStatus* st = nn->api->SessionOptionsAppendExecutionProvider_CUDA(nn->opts, &cuda_opts);
        if (st != NULL) {
            const char* msg = nn->api->GetErrorMessage(st);
            fprintf(stderr, "warn: CUDA EP not available (%s); falling back to CPU\n",
                    msg ? msg : "(null)");
            nn->api->ReleaseStatus(st);
        }
    }

    ORT_CHECK(nn->api->CreateSession(nn->env, onnx_path, nn->opts, &nn->session));
    ORT_CHECK(nn->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &nn->mem_info));
    ORT_CHECK(nn->api->GetAllocatorWithDefaultOptions(&nn->alloc));

    /* Cache input/output names so we don't realloc per call. */
    char* name = NULL;
    ORT_CHECK(nn->api->SessionGetInputName(nn->session, 0, nn->alloc, &name));
    nn->input_name = name;
    ORT_CHECK(nn->api->SessionGetOutputName(nn->session, 0, nn->alloc, &name));
    nn->policy_name = name;
    ORT_CHECK(nn->api->SessionGetOutputName(nn->session, 1, nn->alloc, &name));
    nn->value_name = name;

    *out = nn;
    return 0;
}

void nn_free(NN* nn) {
    if (!nn) return;
    if (nn->input_name)  nn->alloc->Free(nn->alloc, nn->input_name);
    if (nn->policy_name) nn->alloc->Free(nn->alloc, nn->policy_name);
    if (nn->value_name)  nn->alloc->Free(nn->alloc, nn->value_name);
    if (nn->mem_info)    nn->api->ReleaseMemoryInfo(nn->mem_info);
    if (nn->session)     nn->api->ReleaseSession(nn->session);
    if (nn->opts)        nn->api->ReleaseSessionOptions(nn->opts);
    if (nn->env)         nn->api->ReleaseEnv(nn->env);
    free(nn);
}

void nn_board_to_input(const Board* b, float* out_plane) {
    /* Layout: (1, N, N) float, indexed by (x, y) just like the Python
     * canonical-form numpy board. +1 for own, -1 for opp, 0 for empty. */
    for (int i = 0; i < OTHELLO_NN; i++) {
        uint64_t bit = 1ULL << i;
        if      (b->p0 & bit) out_plane[i] = +1.0f;
        else if (b->p1 & bit) out_plane[i] = -1.0f;
        else                  out_plane[i] =  0.0f;
    }
}

void nn_predict_batch(NN* nn, const float* boards_NCHW, int B,
                      float* pis, float* vs) {
    int64_t in_shape[4] = { (int64_t)B, 1, OTHELLO_N, OTHELLO_N };

    OrtValue* in = NULL;
    OrtStatus* st = nn->api->CreateTensorWithDataAsOrtValue(
        nn->mem_info, (void*)boards_NCHW,
        (size_t)B * OTHELLO_NN * sizeof(float),
        in_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in);
    if (st != NULL) {
        fprintf(stderr, "ORT CreateTensor failed: %s\n", nn->api->GetErrorMessage(st));
        nn->api->ReleaseStatus(st);
        abort();
    }

    const char* input_names[1]  = { nn->input_name };
    const char* output_names[2] = { nn->policy_name, nn->value_name };
    OrtValue* outs[2] = { NULL, NULL };

    st = nn->api->Run(nn->session, NULL,
                      input_names, (const OrtValue* const*)&in, 1,
                      output_names, 2, outs);
    if (st != NULL) {
        fprintf(stderr, "ORT Run failed: %s\n", nn->api->GetErrorMessage(st));
        nn->api->ReleaseStatus(st);
        abort();
    }

    /* Copy out policy log-probs and exp() in place; copy out value scalars. */
    float* p_out = NULL;
    float* v_out = NULL;
    OrtStatus* st0 = nn->api->GetTensorMutableData(outs[0], (void**)&p_out);
    OrtStatus* st1 = nn->api->GetTensorMutableData(outs[1], (void**)&v_out);
    if (st0 != NULL || st1 != NULL) {
        if (st0) nn->api->ReleaseStatus(st0);
        if (st1) nn->api->ReleaseStatus(st1);
        fprintf(stderr, "ORT GetTensorMutableData failed\n");
        abort();
    }

    /* policy: (B, ACTION_SIZE) log-softmax -> exp */
    for (int i = 0; i < B * OTHELLO_ACTION_SIZE; i++) {
        pis[i] = expf(p_out[i]);
    }
    /* value: (B, 1) -> (B,) */
    for (int i = 0; i < B; i++) vs[i] = v_out[i];

    nn->api->ReleaseValue(in);
    nn->api->ReleaseValue(outs[0]);
    nn->api->ReleaseValue(outs[1]);
}
