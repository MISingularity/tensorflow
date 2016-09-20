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

//Status LoadGraph(string graph_file_name, tensorflow::Session** session) {
//  GraphDef graph_def;
//  TF_RETURN_IF_ERROR(
//      ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
//  TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
//  TF_RETURN_IF_ERROR((*session)->Create(graph_def));
//  return Status::OK();
//}
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

static std::vector<int> inference(const std::vector<int>& pinyin_str, std::unique_ptr<tensorflow::Session> *session) {
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
  const int T = 25;
  tensorflow::Tensor input_tensor(
        tensorflow::DT_INT32,
        tensorflow::TensorShape({2, 1, 25}));
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
  s = (*session)->Run(inputs, output_tensor_names, {}, &output_tensors);
  LOG(INFO) << "phase5";
  tensorflow::Tensor* ans = &output_tensors[0];
  LOG(INFO) << "phase6";
  LOG(INFO) << ans;
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

int main(int argc, char** argv) {
  // Construct your graph.
  tensorflow::GraphDef graph;
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  TF_CHECK_OK(root.ToGraphDef(&graph));

  // Create a Session running TensorFlow locally in process.
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));

//  const char* const model_cstr = env->GetStringUTFChars(model, NULL);
//  const char* const ch_labels_cstr = env->GetStringUTFChars(chinese_labels, NULL);
//  const char* const py_labels_cstr = env->GetStringUTFChars(pinyin_labels, NULL);
  char* py_labels_sctr = "tensorflow/cc/speech/pinyin.txt";
  char* ch_labels_sctr = "tensorflow/cc/speech/chinese.txt";
  LOG(INFO) << "reading file starts";
  std::vector<std::string> pinyin_labels = ReadFileToVector(py_labels_sctr);
  std::vector<std::string> ch_labels = ReadFileToVector(ch_labels_sctr);
  LOG(INFO) << ch_labels.size();
  LOG(INFO) << ch_labels[0];
  LOG(INFO) << ch_labels[ch_labels.size() - 1];
  LOG(INFO) << "reading file finished";
  clm::WordsDictionary* py_dict_ = new clm::WordsDictionary(pinyin_labels);
  clm::WordsDictionary* ch_dict_ = new clm::WordsDictionary(ch_labels);

  LOG(INFO) << "building dic finished";
  std::string res = "wo shi tian cai";
  tensorflow::Status s = LoadGraph("tensorflow/cc/speech/graph.pb", &session);
  LOG(INFO) << "loading graph finished";
  std::vector<std::string> words = split(res, ' ');
  std::vector<int> pinyin_str;
  int T = 25;
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
  std::vector<int> ids = inference(pinyin_str, &session);
  LOG(INFO) << "phase9";
  std::string ret = "";
//  LOG(INFO) << ch_dict_.size();
  for (int id : ids) {
    ret += ch_dict_->getWordById(id) + " ";
  }
  LOG(INFO) << ret.c_str();
  LOG(INFO) << "finished";


  return 0;
}

