# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

py_library(
    name = "lookahead_py",
    srcs = [
        "__init__.py",
        "lookahead_ops.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "lookahead_op_test",
    size = "small",
    srcs = ["lookahead_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lookahead_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "lookahead_grad_op_test",
    size = "small",
    srcs = ["lookahead_grad_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lookahead_py",
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
