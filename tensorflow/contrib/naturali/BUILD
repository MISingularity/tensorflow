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
    "tf_kernel_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

py_library(
    name = "naturali_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]) + glob(["tools/*.py"]),
    data = [
        ":python/ops/_lookahead_ops.so",
        ":python/ops/_octbit_ops.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

tf_custom_op_library(
    name = "python/ops/_lookahead_ops.so",
    srcs = [
        "kernels/lookahead_ops.cc",
        "kernels/lookahead_grad_ops.cc",
        "ops/lookahead_ops.cc",
    ],
    gpu_srcs = [
        "kernels/lookahead_ops.cu.cc",
        "kernels/lookahead_grad_ops.cu.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)

tf_custom_op_library(
    name = "python/ops/_octbit_ops.so",
    srcs = [
        "kernels/octbit_mat_mul_op.cc",
        "ops/octbit_ops.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)

tf_cc_test(
    name = "octbit_mat_mul_test",
    size = "small", 
    srcs = ["kernels/octbit_mat_mul_op_test.cc"],
    deps = [
        ":octbit_ops",
		":cc_octbit_ops", 
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "octbit_ops",
    srcs = [
        "kernels/octbit_mat_mul_op.cc",
    ],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:concat_lib_hdrs",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:pooling_ops",
        "//third_party/eigen3",
        "@gemmlowp//:eight_bit_int_gemm",
    ],
)

cc_library(
    name = "cc_octbit_ops",
    srcs = ["ops/octbit_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)


cuda_py_tests(
    name = "lookahead_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/lookahead_ops_test.py"],
    additional_deps = [
        ":naturali_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)


filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
