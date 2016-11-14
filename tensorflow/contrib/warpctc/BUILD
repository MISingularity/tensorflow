# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_libraries",
    "tf_kernel_library",
    "cc_header_only_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)

py_library(
    name = "warpctc_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    data = [
        ":python/ops/_warpctc_ops.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

tf_custom_op_library(
    name = "python/ops/_warpctc_ops.so",
    srcs = [
        "kernels/warpctc_ops.cc",
	"kernels/warp-ctc/include/ctc.h",
        "ops/warpctc_ops.cc",
        "kernels/warp-ctc/tests/test.h",
    ],
    gpu_srcs = [
        "kernels/warpctc_ops.cu.cc",
        "kernels/warpctc_ops.h",
        "kernels/warp-ctc/include/ctc.h",
        "kernels/warp-ctc/src/ctc_entrypoint.cu.cc",
        "kernels/warp-ctc/src/reduce.cu.cc",
        "kernels/warp-ctc/tests/test.h",
    ], 
    deps =
    [
        "//tensorflow/contrib/warpctc/kernels/warp-ctc/include/detail:libcpu",
        "//tensorflow/contrib/warpctc/kernels/warp-ctc/include/detail:libgpu",
        "//tensorflow/contrib/warpctc/kernels/warp-ctc/include/contrib/moderngpu/include:lib_gpu",
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)

cuda_py_tests(
    name = "warpctc_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/warpctc_ops_test.py"],
    additional_deps = [
        ":warpctc_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)
