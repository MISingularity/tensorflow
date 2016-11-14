/* Copyright 2016 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/ctc_ops.cc.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "tensorflow/contrib/warpctc/kernels/warp-ctc/include/detail/ctc_helper.h"

namespace tensorflow {
// Used for easy access to 3d tensor.
template<typename T>
class Accessor3D {
 public:
  Accessor3D(int pb, int pf, T* pv) : b(pb), f(pf), values(pv) {}

  T operator()(int ti, int bi, int fi) const {
    return values[(ti*b + bi)*f + fi];
  }

  T& operator()(int ti, int bi, int fi) {
    return values[(ti*b + bi)*f + fi];
  }

 private:
  int b, f;
  // not owner.
  T* values;
};

template<typename T>
class Accessor2D {
 public:
  Accessor2D(int pf, T* pv) : f(pf), values(pv) {}

  T operator()(int bi, int fi) const {
    return values[bi*f + fi];
  }

  T& operator()(int bi, int fi) {
    return values[bi*f + fi];
  }

 private:
  int f;
  // not owner.
  T* values;
};

// This a helper class that use for compute for an single sequence, it will be reused for
// each sequence in a minibatch.
template<typename ProbT>
class CpuCTCWorkspace {

 private:
  int setup_labels(const int* const labels, int L, int S, int BLANK) {
    int e_counter = 0;
    int s_counter = 0;

    s_inc[s_counter++] = 1;

    int repeats = 0;

    for (int i = 1; i < L; ++i) {
      if (labels[i-1] == labels[i]) {
        s_inc[s_counter++] = 1;
        s_inc[s_counter++] = 1;
        e_inc[e_counter++] = 1;
        e_inc[e_counter++] = 1;
        ++repeats;
      }
      else {
        s_inc[s_counter++] = 2;
        e_inc[e_counter++] = 2;
      }
    }
    e_inc[e_counter++] = 1;

    for (int i = 0; i < L; ++i) {
      labels_w_blanks[2 * i] = BLANK;
      labels_w_blanks[2 * i + 1] = labels[i];
    }
    labels_w_blanks[S - 1] = BLANK;

    return repeats;
  }

 public:
  ~CpuCTCWorkspace() {
    delete[] alphas;
    delete[] betas;
    delete[] labels_w_blanks;
    delete[] e_inc;
    delete[] s_inc;
    delete[] output;
  }

  CpuCTCWorkspace(int L, int S, int T, int a) : repeats(0), alphabet_size(a){
    alphas = new ProbT[S*T];
    betas = new ProbT[S];

    labels_w_blanks = new int[S];
    e_inc = new int[S];
    s_inc = new int[S];

    output = new ProbT[alphabet_size];
  }

  void reset(int L, int S, int T, const int* const labels) {
    std::fill(alphas, alphas + S * T, ctc_helper::neg_inf<ProbT>());
    std::fill(betas, betas + S, ctc_helper::neg_inf<ProbT>());
    repeats = setup_labels(labels, L, S, alphabet_size - 1);
  }

  ProbT* alphas;
  ProbT* betas;
  int* labels_w_blanks;
  int* e_inc;
  int* s_inc;
  ProbT* output;
  int repeats;
  const int alphabet_size;
};

template<typename ProbT, bool allow_skip_blank = true>
class CpuCTC {
 public:
  ~CpuCTC() {
    delete workspace_;
    delete[] probs;
  }

  // Noncopyable
  CpuCTC(int alphabet_size, int minibatch,
         const ProbT* const pactivations,
         const int* const pflat_labels,
         const int* const plabel_lengths,
         const int* const pinput_lengths,
         ProbT* pcosts, ProbT* pgradients) :
    alphabet_size_(alphabet_size), blank_(alphabet_size-1),
    minibatch_(minibatch), workspace_(nullptr),
    activations(pactivations), flat_labels(pflat_labels), label_lengths(plabel_lengths),
    input_lengths(pinput_lengths), costs(pcosts), grads(pgradients),
    probs(nullptr) {
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);
    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);;
    int maxS = 2 * maxL + 1;
    workspace_ = new CpuCTCWorkspace<ProbT>(maxL, maxS, maxT, alphabet_size_);

    // compute softmax, need to make sure the order is right: currently it is t x b x f
    probs = new ProbT[maxT * minibatch_ * alphabet_size_];
  };

  CpuCTC(const CpuCTC&) = delete;
  CpuCTC& operator=(const CpuCTC&) = delete;

  void cost_and_grad() {
    // compute softmax, need to make sure the order is right: currently it is t x b x f
    for (int mb = 0; mb < minibatch_; ++mb) {
      for(int c = 0; c < input_lengths[mb]; ++c) {
        int col_offset = (mb + minibatch_ * c) * alphabet_size_;
        ProbT max_activation = -std::numeric_limits<ProbT>::infinity();
        for(int r = 0; r < alphabet_size_; ++r)
          max_activation = std::max(max_activation, activations[r + col_offset]);

        ProbT denom = ProbT(0.);
        for(int r = 0; r < alphabet_size_; ++r) {
          probs[r + col_offset] = std::exp(activations[r + col_offset] - max_activation);
          denom += probs[r + col_offset];
        }

        for(int r = 0; r < alphabet_size_; ++r) {
          probs[r + col_offset] /= denom;
        }
      }
    }

    for (int mb = 0; mb < minibatch_; ++mb) {
      const int T = input_lengths[mb]; // Length of utterance (time)
      const int L = label_lengths[mb]; // Number of labels in transcription

      bool mb_status;
      int label_count = std::accumulate(label_lengths, label_lengths + mb, 0);
      std::tie(costs[mb], mb_status) =
          cost_and_grad_kernel(flat_labels + label_count, T, L, mb);
    }
  }

 private:
  ctc_helper::log_plus<ProbT> log_add;

  int alphabet_size_; // Number of characters plus blank
  int blank_;
  int minibatch_;
  CpuCTCWorkspace<ProbT>* workspace_;

  const ProbT* const activations;
  const int* const flat_labels;
  const int* const label_lengths;
  const int* const input_lengths;

  ProbT* costs;
  ProbT* grads;

  ProbT* probs;

  std::tuple<ProbT, bool>
  cost_and_grad_kernel(const int* const labels, int T, int L, int mb) {
    const int S = 2*L + 1; // Number of labels with blanks

    workspace_->reset(L, S, T, labels);
    CpuCTCWorkspace<ProbT>& ctcm(*workspace_);

    bool over_threshold = false;

    // T should be at least greater than L + number of repeats.
    CHECK(L + ctcm.repeats <= T);

    ProbT llForward = compute_alphas(S, T, mb, ctcm);

    ProbT llBackward = compute_betas_and_grad(llForward, S, T, mb, ctcm);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
      over_threshold = true;
    }

    return std::make_tuple(-llForward, over_threshold);
  }


  ProbT compute_alphas(int S, int T, int mb, CpuCTCWorkspace<ProbT>& ctcm) {
    int repeats = ctcm.repeats;
    const int* const e_inc = ctcm.e_inc;
    const int* const s_inc = ctcm.s_inc;
    const int* const labels = ctcm.labels_w_blanks;

    Accessor2D<ProbT> alphas(alphabet_size_, ctcm.alphas);
    Accessor3D<ProbT> yprobs(minibatch_, alphabet_size_, probs);

    int start =  0;
    int end = S > 1 ? 2 : 1;

    // Setup the boundary condition.
    alphas(0, 0) = std::log(yprobs(0, mb, labels[0]));  // should be blank
    alphas(0, 1) = std::log(yprobs(0, mb, labels[1]));  // first payload

    for(int t = 1; t < T; ++t) {
      // Still a bit murky here.
      int remain = (S / 2) + repeats - (T - t);
      if(remain >= 0) start += s_inc[remain];
      if(t <= (S / 2) + repeats) end += e_inc[t - 1];
      int startloop = start;

      if (start == 0) {
        alphas(t, 0) = alphas(t-1, 0) + std::log(yprobs(t, mb, blank_));
        startloop += 1;
      }

      for(int i = startloop; i < end; ++i) {
        ProbT prev_sum = log_add(alphas(t - 1, i), alphas(t, i - 1));

        // Skip two if not on blank and not on repeat.
        if (allow_skip_blank && labels[i] != blank_ && i >= 2 && labels[i] != labels[i-2])
          prev_sum = log_add(prev_sum, alphas(t - 1, i - 2));

        alphas(t, i) = prev_sum + std::log(yprobs(t, mb, labels[i]));
      }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
      loglike = log_add(loglike, alphas(T- 1, i));
    }

    return loglike;
  }


  // Starting from T, we sweep backward over the alpha array computing one column
  // of betas as we go.  At each position we can update product alpha * beta and then
  // sum into the gradient associated with each label.
  // NOTE computes gradient w.r.t UNNORMALIZED final layer activations.
  // Assumed passed in grads are already zeroed!
  ProbT compute_betas_and_grad(ProbT log_partition, int S, int T, int mb,
                               CpuCTCWorkspace<ProbT>& ctcm) {
    int repeats = ctcm.repeats;
    const int* const e_inc = ctcm.e_inc;
    const int* const s_inc = ctcm.s_inc;
    const int* const labels = ctcm.labels_w_blanks;

    Accessor2D<ProbT> alphas(alphabet_size_, ctcm.alphas);
    Accessor3D<ProbT> yprobs(minibatch_, alphabet_size_, probs);
    Accessor3D<ProbT> ygrads(minibatch_, alphabet_size_, grads);

    ProbT* betas = ctcm.betas;
    ProbT* output = ctcm.output;

    int start = S > 1 ? (S - 2) : 0;
    int end = (T > (S / 2) + repeats) ? S : S-1;

    std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

    //set the starting values in the beta column at the very right edge
    for (int i = start; i < end; ++i) {
      betas[i] = std::log(yprobs(T -1, mb, labels[i]));

      //compute alpha * beta in log space at this position in (S, T) space
      alphas(T -1, i) += betas[i];

      //update the gradient associated with this label
      //essentially performing a reduce-by-key in a sequential manner
      output[labels[i]] = log_add(alphas(T -1, i), output[labels[i]]);
    }

    //update the gradient wrt to each unique label
    for (int i = 0; i < alphabet_size_; ++i) {
      if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
          yprobs(T - 1, mb, i) == 0.0) {
        ygrads(T - 1, mb, i) = yprobs(T - 1, mb, i);
      } else {
        ygrads(T - 1, mb, i) = yprobs(T - 1, mb, i)
                  - std::exp(output[i] - std::log(yprobs(T - 1, mb, i)) - log_partition);
      }
    }

    //loop from the second to last column all the way to the left
    for(int t = T - 2; t >= 0; --t) {
      int remain = (S / 2) + repeats - (T - t);
      if(remain >= -1) start -= s_inc[remain + 1];
      if(t < (S / 2) + repeats) end -= e_inc[t];
      int endloop = end == S ? end - 1 : end;

      std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

      for(int i = start; i < endloop; ++i) {
        ProbT next_sum = log_add(betas[i], betas[i+1]);
        // Skip two if not on blank and not on repeat.
        if (labels[i] != blank_ && i != (S-2) && labels[i] != labels[i+2]){
          next_sum = log_add(next_sum, betas[i+2]);
        }
        betas[i] = next_sum + std::log(yprobs(t, mb, labels[i]));

        //compute alpha * beta in log space
        alphas(t, i) += betas[i];

        //update the gradient associated with this label
        output[labels[i]] = log_add(alphas(t, i), output[labels[i]]);
      }

      if (end == S) {
        betas[S-1] = betas[S-1] + std::log(yprobs(t, mb, blank_));
        alphas(t, S-1) += betas[S-1];

        output[labels[S-1]] = log_add(alphas(t, S-1), output[labels[S-1]]);
      }

      // go over the unique labels and compute the final grad
      // wrt to each one at this time step
      for (int i = 0; i < alphabet_size_; ++i) {
        if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
            yprobs(t, mb, i) == 0.0) {
          ygrads(t, mb, i) = yprobs(t, mb, i);
        } else {
          ygrads(t, mb, i) =
              yprobs(t, mb, i) - std::exp(output[i] - std::log(yprobs(t, mb, i)) - log_partition);
        }
      }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
      loglike = log_add(loglike, betas[i]);
    }

    return loglike;
  }
};

class CpuWarpCTCLossOp : public OpKernel {
 public:
  explicit CpuWarpCTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool p, c;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated", &p));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctc_merge_repeated", &c));
  }

  void Compute(OpKernelContext* ctx) override {
    /* activations pointer to the activations in either CPU or GPU
    *  addressable memory, depending on info.  We assume a fixed
    *  memory layout for this 3 dimensional tensor, which has dimension
    *  (t, n, p), where t is the time index, n is the minibatch index,
    *  and p indexes over probabilities of each symbol in the alphabet.
    *  More precisely, element (t, n, p), for a problem with mini_batch examples
    *  in the mini batch, and alphabet_size symbols in the alphabet, is located at:
    *  activations[(t * mini_batch + n) * alphabet_size + p]
    */
    const Tensor& input_tensor = ctx->input(0);

    const Tensor& labels_indices_tensor = ctx->input(1);
    const Tensor& labels_values_tensor = ctx->input(2);
    const Tensor& seq_len_tensor = ctx->input(3);

    auto input = input_tensor.tensor<float, 3>();

    float *activations = (float*)input.data();
    const TensorShape& inputs_shape = input_tensor.shape();

    const int64 max_time = inputs_shape.dim_size(0);
    const int64 batch_size = inputs_shape.dim_size(1);
    const int64 num_classes = inputs_shape.dim_size(2);

    TensorShape labels_shape({batch_size, max_time});
    std::vector<int64> order{0, 1};
    auto labels_indices = labels_indices_tensor.tensor<int64, 2>();
    auto labels_values = labels_values_tensor.tensor<int, 1>();
    auto seq_len = seq_len_tensor.tensor<int, 1>();

    // get the label and label_length ready.
    int64 last_first = 0, sum = 0;
    //  A concatenation of all the labels for the minibatch.
    std::vector<int> labels;
    // The length of each label for each example in the minibatch.
    std::vector<int> label_lengths;
    for(int i = 0; i < labels_indices_tensor.dim_size(0); i++) {
      int64 first;
      int label_of_first;
      first = labels_indices(i,0);
      label_of_first = labels_values(i);
      labels.push_back(label_of_first);
      if (first == last_first) {
        sum++;
      } else {
        label_lengths.push_back(sum);
        last_first = first;
        sum = 1;
      }
    }
    if (sum != 0) {
      label_lengths.push_back(sum);
    }

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len_tensor.shape(), &loss));
    auto loss_t = loss->vec<float>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<float, 3>();

    CpuCTC<float> ctc(num_classes, seq_len_tensor.dim_size(0), activations,
                      labels.data(), label_lengths.data(),
                      seq_len.data(), loss_t.data(), gradient_t.data());

    ctc.cost_and_grad();
  }
};

REGISTER_KERNEL_BUILDER(Name("WarpCtcLoss").Device(DEVICE_CPU), CpuWarpCTCLossOp);
}  // end namespace tensorflow
