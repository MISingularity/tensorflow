#include <cstdio>
#include <functional>
#include <string>
#include <vector>

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

using tensorflow::string;
using tensorflow::int32;

int main(int argc, char** argv) {
  // Construct your graph.
  tensorflow::GraphDef graph;
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  auto a = Const<float>(root, {{3, 2}, {-1, 0}});

  auto x = Const(root.WithOpName("x"), {{1.f}, {1.f}});

  auto y = MatMul(root.WithOpName("y"), a, x);

  TF_CHECK_OK(root.ToGraphDef(&graph));

  // Create a Session running TensorFlow locally in process.
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));

  // Initialize the session with the graph.
  tensorflow::Status s = session->Create(graph);
  if (!s.ok()) {
    std::printf("nothing");
  }
  graph.Clear();
  // Specify the 'feeds' of your network if needed.
  // std::vector<std::pair<string, tensorflow::Tensor>> inputs;

  // Run the session, asking for the first output of "my_output".
  std::vector<tensorflow::Tensor> outputs;
  s = session->Run({}, {"y:0"}, {}, &outputs);
  auto outer = outputs[0].shaped<float, 2>({2, 1});
  // Close the session.
  std::cout << outer;
  session->Close();

  return 0;
}

