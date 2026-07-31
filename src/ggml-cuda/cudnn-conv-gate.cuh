#pragma once

#include <cstdlib>
#include <cstring>

// Value-honouring gate for the cuDNN conv borrows.
//
// 🔴 WHY THIS FILE EXISTS. Every one of these gates used to be a bare `getenv("...")`, i.e. a
// PRESENCE test. `GGML_CUDNN_CONV=0` therefore ENABLED the borrow — the exact opposite of what
// anyone writing it means, and silently: the run is correct, just not the run you asked for. Any
// A/B that "disabled cuDNN conv" that way measured cuDNN conv twice. Two of the three fleet
// services set one of these vars explicitly, so the wrong reading was live.
//
// The tri-state matters because the two vars cross-enable each other (a GGML_OP_CONV_2D can be
// emitted by a service that only asked for CONV3D, and vice versa — see the comment in
// ggml_cuda_op_conv2d_cudnn). So "unset" cannot mean the same thing as "explicitly 0":
//
//   unset          this var has no opinion; the sibling var may still enable the borrow
//   "0" / "false"  this var VETOES its own dimension, whatever the sibling says
//   anything else  enables
//
// That keeps every current deployment bit-identical (nobody sets a 0 today) while making the 0
// that was always intended actually work.

// -1 = unset/empty, 0 = explicitly off, 1 = explicitly on.
static inline int ggml_cudnn_env_tristate(const char * name) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return -1;
    }
    if (strcmp(e, "0") == 0 || strcmp(e, "false") == 0 || strcmp(e, "FALSE") == 0 ||
        strcmp(e, "off") == 0 || strcmp(e, "OFF") == 0) {
        return 0;
    }
    return 1;
}

// `self` vetoes; either var enables. Order of the two arguments is "my var first".
static inline bool ggml_cudnn_conv_gate(const char * self, const char * sibling) {
    const int s = ggml_cudnn_env_tristate(self);
    if (s == 0) {
        return false;
    }
    return s == 1 || ggml_cudnn_env_tristate(sibling) == 1;
}

static inline bool ggml_cudnn_conv2d_enabled() {
    return ggml_cudnn_conv_gate("GGML_CUDNN_CONV", "GGML_CUDNN_CONV3D");
}

static inline bool ggml_cudnn_conv3d_enabled() {
    return ggml_cudnn_conv_gate("GGML_CUDNN_CONV3D", "GGML_CUDNN_CONV");
}
