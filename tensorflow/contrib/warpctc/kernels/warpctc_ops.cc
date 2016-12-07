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
#include "tensorflow/contrib/warpctc/kernels/ctc_helper.h"


namespace tensorflow {

template<typename ProbT>
class CpuCTCWorkspace {
 private:
  void setup_labels(int mb, const int* plabels, int L, int S, int BLANK) {

    int base_2d = mb * L_size;
    for (int i = 0; i < L; ++i) {
      labels[i + base_2d] = plabels[i];
    }


    int l = 1;
    slot_start[base_2d] = 0;
    slot_end[base_2d] = 1;
    slot_index[base_2d] = 0;
    int last_i = 1;
    for (int i = 1; i < L; ++i) {
      if (labels[base_2d + i] == BLANK) {
        slot_start[base_2d + l] = last_i;
        slot_end[base_2d + l] = i;
        ++l;
        slot_start[base_2d + l] = i;
        slot_end[base_2d + l] = i + 1;
        // blank setting
        slot_index[base_2d + i] = l;
        ++l;
        last_i = i + 1;
      } else {
        slot_index[base_2d + i] = l;
        // non-blank  setting
      }
    }

    int e_counter = base_2d;
    int s_counter = base_2d;
    s_inc[s_counter++] = 1;
    for (int i = 1; i < S; ++i) {
      s_inc[s_counter++] = slot_end[base_2d + i * 2 - 1] - slot_start[base_2d + i * 2 - 1];
      s_inc[s_counter++] = 1;
      e_inc[e_counter++] = 1;
      e_inc[e_counter++] = slot_end[base_2d + i * 2 + 1] - slot_start[base_2d + i * 2 + 1];
    }
    e_inc[e_counter++] = 1;
    return;
  }

 public:
  ~CpuCTCWorkspace() {
    delete[] alphas;
    delete[] betas;
    delete[] e_inc;
    delete[] s_inc;
    delete[] slot_index;
    delete[] slot_end;
    delete[] slot_start;
    delete[] output;
    delete[] L_arr;
    delete[] S_arr;
    delete[] T_arr;
    delete[] labels;
  }

  CpuCTCWorkspace(int B, int L, int T, int a, int blank_index)
    :  L_size(L), T_size(T), alphabet_size(a) {
    alphas = new ProbT[B*L*T];
    betas = new ProbT[B*L];

    labels = new int[B*L];
    L_arr = new int[B];
    S_arr = new int[B];
    T_arr = new int[B];

    e_inc = new int[B*L];
    s_inc = new int[B*L];

    slot_index = new int[B*L];

    slot_end = new int[B*L];
    slot_start = new int[B*L];


    output = new ProbT[B*alphabet_size];
    blank_index_ = blank_index;
  }

  void reset(int mb, int L, int S, int T, const int* const labels) {
    std::fill(alphas + mb * L_size * T, alphas + (mb + 1) * L_size * T, ctc_helper::neg_inf<ProbT>());
    std::fill(betas + mb * L_size, betas + (mb + 1) * L_size, ctc_helper::neg_inf<ProbT>());
    L_arr[mb] = L;
    S_arr[mb] = S;
    T_arr[mb] = T;
    setup_labels(mb, labels, L, S, blank_index_);
  }

  ProbT* alphas;
  ProbT* betas;
  int* labels;

  int* slot_index;
  int* slot_end;
  int* slot_start;
  int* e_inc;
  int* s_inc;
  int* L_arr;
  int* T_arr;
  int* S_arr;


  ProbT* output;
  int blank_index_;
  const int L_size;
  const int T_size;
  const int alphabet_size;
};

template<typename ProbT, bool allow_skip_blank = true>
class CpuCTC {
 public:
  ~CpuCTC() {
    delete[] probs;
  }

  // Noncopyable
  CpuCTC(int alphabet_size, int minibatch,
         const ProbT* const pactivations,
         CpuCTCWorkspace<ProbT>* ws,
         const int* const pinput_lengths,
         ProbT* pcosts, ProbT* pgradients,
         const int blank_index = 0) :
    alphabet_size_(alphabet_size), blank_index_(blank_index),
    minibatch_(minibatch), workspace_(ws),
    activations(pactivations), input_lengths(pinput_lengths),
    costs(pcosts), grads(pgradients), probs(nullptr) {
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch);
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
      costs[mb] = cost_and_grad_kernel(mb);
    }
  }

 private:
  ctc_helper::log_plus<ProbT> log_add;

  int alphabet_size_; // Number of characters plus blank
  int blank_index_;
  int minibatch_;
  CpuCTCWorkspace<ProbT>* workspace_;

  const ProbT* const activations;
  const int* const input_lengths;

  ProbT* costs;
  ProbT* grads;

  ProbT* probs;

  ProbT cost_and_grad_kernel(int mb) {
    bool over_threshold = false;

    // T should be at least greater than L + number of repeats
    ProbT llForward = compute_alphas(mb);
    ProbT llBackward = compute_betas_and_grad(llForward, mb);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
      over_threshold = true;
    }

    return -llForward;
  }


  ProbT compute_alphas(int mb) {
    CpuCTCWorkspace<ProbT>& ctcm(*workspace_);
    int start_2d = mb * ctcm.L_size;
    int start_3d = mb * ctcm.L_size * ctcm.T_size;
    const int T = ctcm.T_arr[mb];
    const int L = ctcm.L_arr[mb];
    const int S = ctcm.S_arr[mb];
    const int* const e_inc = ctcm.e_inc + start_2d;
    const int* const s_inc = ctcm.s_inc + start_2d;
    const int* const labels = ctcm.labels + start_2d;
    const int* const slot_end = ctcm.slot_end + start_2d;
    const int* const slot_start = ctcm.slot_start + start_2d;
    const int* const slot_index = ctcm.slot_index + start_2d;

    Accessor2D<ProbT> alphas(ctcm.L_size, ctcm.alphas + start_3d);
    Accessor3D<ProbT> yprobs(minibatch_, alphabet_size_, probs);
    int least_slot_num = 2 * S - 1;
    int start = ((least_slot_num - T) < 0) ? 0 : 1;
    int end = S > 0 ? slot_end[1] : 1;
    for (int i = start; i < end; ++i) {
        alphas(0, i) = std::log(yprobs(0, mb, labels[i]));
    }
    for(int t = 1; t < T; ++t) {
      // Still a bit murky here.
      int remain = least_slot_num - (T - t);
      if(remain >= 0) start += s_inc[remain];
      if(t <= least_slot_num) end += e_inc[t - 1];
      int startloop = start;

      if (start == 0) {
        alphas(t, 0) = alphas(t-1, 0) + std::log(yprobs(t, mb, blank_index_));
        startloop += 1;
      }

      for(int i = startloop; i < end; ++i) {
        int l = slot_index[i];
        ProbT prev_sum = alphas(t - 1, i);
        for (int j = slot_start[l - 1]; j < slot_end[l - 1]; j++) {
           prev_sum = log_add(prev_sum, alphas(t - 1, j));
        }
        // Skip two if not on blank and not on repeat.
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
  ProbT compute_betas_and_grad(ProbT log_partition, int mb) {
    CpuCTCWorkspace<ProbT>& ctcm(*workspace_);
    int start_2d = mb * ctcm.L_size;
    int start_3d = mb * ctcm.L_size * ctcm.T_size;
    const int T = ctcm.T_arr[mb];
    const int S = ctcm.S_arr[mb];
    const int L = ctcm.L_arr[mb];
    const int* const e_inc = ctcm.e_inc + start_2d;
    const int* const s_inc = ctcm.s_inc + start_2d;
    const int* const labels = ctcm.labels + start_2d;
    const int* const slot_end = ctcm.slot_end + start_2d;
    const int* const slot_start = ctcm.slot_start + start_2d;
    const int* const slot_index = ctcm.slot_index + start_2d;


    Accessor2D<ProbT> alphas(ctcm.L_size, ctcm.alphas + start_3d);
    Accessor3D<ProbT> yprobs(minibatch_, alphabet_size_, probs);
    Accessor3D<ProbT> ygrads(minibatch_, alphabet_size_, grads);

    ProbT* betas = ctcm.betas + start_2d;
    ProbT* output = ctcm.output + mb * ctcm.alphabet_size;

    int least_slot_num = 2 * S  - 1;
    int start = L > 1 ? slot_start[slot_index[L - 2]] : 0;
    int end = (T > least_slot_num) ? L : L-1;

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
      int remain = least_slot_num  - (T - t);
      if(remain >= -1) start -= s_inc[remain + 1];
      if(t < least_slot_num) end -= e_inc[t];
      int endloop = end == L ? end - 1 : end;

      std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());
      for(int i = start; i < endloop; ++i) {
        int l = slot_index[i];
        ProbT next_sum = betas[i];
        for (int j = slot_start[l + 1]; j < slot_end[l + 1]; j++)
          next_sum = log_add(next_sum, betas[j]);
        // Skip two if not on blank and not on repeat.
        betas[i] = next_sum + std::log(yprobs(t, mb, labels[i]));

        //compute alpha * beta in log space
        alphas(t, i) += betas[i];

        //update the gradient associated with this label
        output[labels[i]] = log_add(alphas(t, i), output[labels[i]]);
      }

      if (end == L) {
        int l = slot_index[L - 1];
        for (int j = slot_start[l]; j < slot_end[l]; j++) {
          betas[j] = betas[j] + std::log(yprobs(t, mb, blank_index_));
          alphas(t, j) += betas[j];
          output[labels[j]] = log_add(alphas(t, j), output[labels[j]]);
        }
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
  int blank_index = -1;
  int new_blank_index = -1;
  explicit CpuWarpCTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool p, c;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated", &p));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("new_blank_index", &new_blank_index));
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
    const int alphabet_size = num_classes;

    TensorShape labels_shape({batch_size, max_time});
    std::vector<int64> order{0, 1};
    auto labels_indices = labels_indices_tensor.tensor<int64, 2>();
    auto labels_values = labels_values_tensor.tensor<int, 1>();
    auto seq_len = seq_len_tensor.tensor<int, 1>();

    // get the label and label_length ready.
    int64 last_first = 0, slot_sum = 0, label_sum = 0, last_second = 0;
    //  A concatenation of all the labels for the minibatch.
    std::vector<int> labels;
    // The length of each label for each example in the minibatch.
    // and add a blank between adjacent labels
    // add blank at the begin and the end of minibatch
    std::vector<int> label_lengths;
    std::vector<int> slot_lengths;
    if (labels_indices_tensor.dim_size(0) > 0) {
      // initialization
      last_first = labels_indices(0, 0);
      last_second = labels_indices(0, 1);
      labels.push_back(labels_values(0));
      slot_sum = 1; label_sum = 1;

      for(int i = 1; i < labels_indices_tensor.dim_size(0); i++) {
        int64 first, second;
        int label_of_first;
        first = labels_indices(i, 0);
        second = labels_indices(i, 1);
        label_of_first = labels_values(i);
        if (first == last_first) {
          if (second != last_second) {
              slot_sum++;
          }
          labels.push_back(label_of_first);
          label_sum++;
        } else {
          label_lengths.push_back(label_sum);
          slot_lengths.push_back(slot_sum/2);
          labels.push_back(label_of_first);
          last_first = first;
          slot_sum = 1;
          label_sum = 1;
        }
        last_second = second;
      }
      if (slot_sum != 0) {
        label_lengths.push_back(label_sum);
        slot_lengths.push_back(slot_sum/2);
      }
    }

    // if new blank index exists,
    // substitue old one into new one.
    int final_blank_index = blank_index;
    if (new_blank_index >= 0) {
        for (int i = 0; i < labels.size(); i++) {
            if (labels[i] == blank_index) {
                labels[i] = new_blank_index;
            }
        }
        final_blank_index = new_blank_index;
    }
    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len_tensor.shape(), &loss));
    auto loss_t = loss->vec<float>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<float, 3>();

    // first setup workspace
    int minibatch = seq_len_tensor.dim_size(0);
    const int* const pflat_labels = labels.data();
    const int* const plabel_lengths = label_lengths.data();
    const int* const pslot_lengths = slot_lengths.data();
    const int* const pinput_lengths = seq_len.data();

    int maxT = *std::max_element(pinput_lengths, pinput_lengths + minibatch);
    int maxL = *std::max_element(plabel_lengths, plabel_lengths + minibatch);
    CpuCTCWorkspace<float> workspace(minibatch, maxL, maxT, alphabet_size, final_blank_index);

    for (int mb = 0; mb < minibatch; ++mb) {
      const int T = pinput_lengths[mb]; // Length of utterance (time)
      const int L = plabel_lengths[mb]; // Length of utterance (time)
      const int S = pslot_lengths[mb]; // Number of labels in transcription

      int label_count = std::accumulate(plabel_lengths, plabel_lengths + mb, 0);
      const int* const labels = pflat_labels + label_count;
      workspace.reset(mb, L, S, T, labels);
    }
    CpuCTC<float> ctc(alphabet_size, minibatch, activations, &workspace,
                      seq_len.data(), loss_t.data(), gradient_t.data(), final_blank_index);
    ctc.cost_and_grad();
  }
};

REGISTER_KERNEL_BUILDER(Name("WarpCtcLoss").Device(DEVICE_CPU), CpuWarpCTCLossOp);
}  // end namespace tensorflow
