#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <fstream>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/speech/pkg/dict.h"
//#include "dict.h"

using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;


std::vector<std::string> ReadFileToVector(char* filename) {
    std::ifstream t;
    t.open(filename);
    std::vector<std::string> v;
    while (!t.eof()) {
        std::string ch;
        t >> ch;
        v.push_back(ch);
    }
    return v;
}

class CLM_Service {
public:
  std::unique_ptr<tensorflow::Session> session;
  static const int T = 25;
  clm::WordsDictionary* ch_dict_;
  clm::WordsDictionary* py_dict_;


  CLM_Service(char *graph_pb_path, char *py_labels_sctr, char *ch_labels_sctr) {
    // Construct your graph.
    tensorflow::GraphDef graph;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();

    TF_CHECK_OK(root.ToGraphDef(&graph));
    session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    LOG(INFO) << "reading file starts";
    std::vector<std::string> pinyin_labels = ReadFileToVector(py_labels_sctr);
    std::vector<std::string> ch_labels = ReadFileToVector(ch_labels_sctr);
    LOG(INFO) << ch_labels.size();
    LOG(INFO) << ch_labels[0];
    LOG(INFO) << ch_labels[ch_labels.size() - 1];
    LOG(INFO) << "reading file finished";
    py_dict_ = new clm::WordsDictionary(pinyin_labels);
    ch_dict_ = new clm::WordsDictionary(ch_labels);

    LOG(INFO) << "building dic finished";
    tensorflow::Status s = LoadGraph(graph_pb_path);
    LOG(INFO) << "loading graph finished";
  }

  std::string ContrainModel(string py_str) {
    std::vector<int> pinyin_str;
    std::vector<std::string> words = split(py_str, ' ');
    for (const std::string& w : words) {
      pinyin_str.push_back(py_dict_->getIdByWord(w));
      if (pinyin_str.size() > T) {
        break;
      }
    }

    if (pinyin_str.size() < T) {
        int remains = T - pinyin_str.size();
        for (int i = 0; i < remains; i++) {
          pinyin_str.push_back(py_dict_->getIdByWord("<null>"));
        }
    }
    std::vector<int> ids = inference(pinyin_str);
    LOG(INFO) << "phase9";
    std::string ret = "";
    for (int id : ids) {
      ret += ch_dict_->getWordById(id) + " ";
    }
    return ret;
  }

  Status LoadGraph(string graph_file_name) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                          graph_file_name, "'");
    }
    session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = session->Create(graph_def);
    if (!session_create_status.ok()) {
      return session_create_status;
    }
    return Status::OK();
  }

  std::vector<int> inference(const std::vector<int>& pinyin_str) {
    std::vector<int> ret;

    tensorflow::Status s;
    std::vector<tensorflow::Tensor> output_tensors;
    std::vector<std::string> output_tensor_names;
    std::vector<std::pair<std::string, tensorflow::Tensor> > inputs;

    // initial state
    output_tensors.clear();
    output_tensor_names.clear();
    inputs.clear();

    LOG(INFO) << "phase1";
    tensorflow::Tensor input_tensor(
          tensorflow::DT_INT32,
          tensorflow::TensorShape({2, 1, T}));
    auto input_tensor_data = input_tensor.tensor<int, 3>();
    LOG(INFO) << "phase2";
    for (int i = 0; i < T; i++) {
      input_tensor_data(0,0,i) = 0;
      input_tensor_data(1,0,i) = pinyin_str[i];
    }
    LOG(INFO) << "phase3";
    inputs.push_back(std::make_pair("model/input_data:0", input_tensor));
    output_tensor_names.push_back("model/word_outputs:0");
    LOG(INFO) << "phase4";
    s = session->Run(inputs, output_tensor_names, {}, &output_tensors);
    LOG(INFO) << "phase5";
    tensorflow::Tensor* ans = &output_tensors[0];
    LOG(INFO) << "phase6";

    auto ch = ans->flat<int64>();
    for (int i = 0; i < T; i++) {
      ret.push_back(ch(i));
    }
    LOG(INFO) << "phase7";
    return ret;
  }

  std::vector<std::string> split(const std::string &text, char sep) {
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != std::string::npos) {
      tokens.push_back(text.substr(start, end - start));
      start = end + 1;
    }
    tokens.push_back(text.substr(start));
    return tokens;
  }
};

int main(int argc, char** argv) {

  CLM_Service clm_service = CLM_Service("tensorflow/cc/speech/graph.pb", "tensorflow/cc/speech/pinyin.txt", "tensorflow/cc/speech/chinese.txt");
  LOG(INFO) << clm_service.ContrainModel("wo shi tian cai");
  return 0;
}
