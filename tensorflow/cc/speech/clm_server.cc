#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/server_context.h>
#include <grpc++/security/server_credentials.h>
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
#include "speech.grpc.pb.h"
//#include "dict.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using speech::CLMInput;
using speech::CLMInputUnit;
using speech::CLMOutput;
using speech::SpeechInput;
using speech::SpeechOutput;
using speech::SpeechOutputUnit;
using std::chrono::system_clock;
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


class CLM_Servicer {
public:
  std::unique_ptr<tensorflow::Session> session;
  // model config
  static const int T = 25;
  static const int B = 20;
  
  clm::WordsDictionary* ch_dict_;
  clm::WordsDictionary* py_dict_;
  tensorflow::GraphDef graph;
  CLM_Servicer() {}

  CLM_Servicer(char *graph_pb_path, char *py_labels_sctr, char *ch_labels_sctr) {
    // Construct your graph.
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

  tensorflow::Status LoadGraph(string graph_file_name) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                          graph_file_name, "'");
    }
    session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = session->Create(graph_def);
    if (!session_create_status.ok()) {
      return session_create_status;
    }
    return tensorflow::Status::OK();
  }

  std::vector<int> inference(const std::vector<std::vector<int>>& pinyin_str) {
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
          tensorflow::TensorShape({2, B, T}));
    auto input_tensor_data = input_tensor.tensor<int, 3>();
    // Now batch_size is only 1
    // If needed, update following code to adopt serveral batch.
    LOG(INFO) << "phase2";
    for (int b = 0; b < B; b++) {
      for (int i = 0; i < T; i++) {
        input_tensor_data(0,b,i) = 0;
        input_tensor_data(1,b,i) = pinyin_str[b][i];
      }
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
    for (int i = 0; i < B*T; i++) {
      ret.push_back(ch(i));
    }
    LOG(INFO) << "phase7";
    return ret;
  }

};

class CLMImpl final : public speech::CLM::Service {
 public:
  char * graph_file_path = "tensorflow/cc/speech/clm_graph.pb";
  char * chinese_dict_path = "tensorflow/cc/speech/chinese.txt";
  char * pinyin_dict_path = "tensorflow/cc/speech/pinyin_without_light_tone.txt";
  CLM_Servicer clm_service;
  explicit CLMImpl() {
     clm_service = CLM_Servicer(graph_file_path, pinyin_dict_path, chinese_dict_path);
  }
  grpc::Status ContrainModel(ServerContext* context, const CLMInput* request, CLMOutput* response) override {
    // Now only support 1 request
    std::vector<std::vector<int>> pinyin_inputs;
    std::vector<int> appendix;
    pinyin_inputs.clear();
    int null_tag = clm_service.py_dict_->getIdByWord("<null>");
    for (int i = 0; i < request->cinput_size(); i++) {
      const CLMInputUnit cinput = request->cinput(i);
      std::vector<int> pinyin_str;
      pinyin_str.clear();
      for (int j = 0; j < cinput.pinyin_list_size(); j++) {
        const ::std::string mpl = cinput.pinyin_list(j);
        std::vector<std::string> words = split(mpl, ' ');
        for (const std::string& w : words) {
          pinyin_str.push_back(clm_service.py_dict_->getIdByWord(w));
          if (pinyin_str.size() > clm_service.T) {
            break;
          }
        }

        if (pinyin_str.size() < clm_service.T) {
            int remains = clm_service.T - pinyin_str.size();
            for (int i = 0; i < remains; i++) {
              pinyin_str.push_back(null_tag);
            }
        }
        pinyin_inputs.push_back(pinyin_str);
        appendix.push_back(i);
      }
    }
    // TODO: if batch_size more than 1, need add padding phase and update the inference/decode part.
    int origin_length = request->cinput_size();
    int diff = clm_service.B - (pinyin_inputs.size() - 1) % clm_service.B - 1;
    std::vector<int> temp;
    for (int i = 0; i < clm_service.T; i++) {
      temp.push_back(null_tag);
    }
    for (int i = 0; i < diff; i++) {
      pinyin_inputs.push_back(temp); 
      appendix.push_back(-1);
    }
    std::vector<std::vector<std::vector<int>>> outputs;
    for (int i = 0; i < origin_length; i++) {
      outputs.push_back(std::vector<std::vector<int>>());
    }
//std::vector<std::vector<int>>())
    LOG(INFO) << pinyin_inputs.size() << " " << origin_length;
    for (int i = 0; i < pinyin_inputs.size() / clm_service.B; i++) { 
      std::vector<std::vector<int>>::const_iterator first = pinyin_inputs.begin() + i*clm_service.B;
      std::vector<std::vector<int>>::const_iterator last = pinyin_inputs.begin() + (i+1)*clm_service.B;
      std::vector<std::vector<int>> newVec(first, last);
      std::vector<int> b_outputs = clm_service.inference(newVec);
      for (int ap_id = i * clm_service.B; ap_id < (i + 1) * clm_service.B; ap_id++) {
        if (appendix[ap_id] < 0) {
          continue;
        }
        int b_id = ap_id - i * clm_service.B;
        std::vector<int>::const_iterator bfirst = b_outputs.begin() + b_id*clm_service.T;
        std::vector<int>::const_iterator blast = b_outputs.begin() + (b_id+1)*clm_service.T;
        std::vector<int> b_item(bfirst, blast);
        outputs[appendix[ap_id]].push_back(b_item);
      } 
    }
    LOG(INFO) << "phase9";
//    for every group of input unit, rerank the items and select the most
//    satisfactory one.
//    Currenly, we only select the first one.
//    Need Weiwan to reimplement it.
    for (int b = 0; b < origin_length; b++) { 
      std::string ret = "";
      for (int id = 0; id < outputs[b].size(); id++) {
        for (int cid = 0; cid < outputs[b][id].size(); cid++) {
          ret += clm_service.ch_dict_->getWordById(outputs[b][id][cid]) + " ";
        }
        break;
      }
      response->add_coutput(ret);
    }
    return grpc::Status::OK;
  }
};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  CLMImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  // Expect only arg: --db_path=path/to/route_guide_db.json.
  RunServer();

  return 0;
}

