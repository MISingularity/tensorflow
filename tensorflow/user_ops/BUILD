# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//tensorflow:internal"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/serving/...",
        "//platforms/techila/...",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_tests")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_libraries",
    "tf_kernel_library",
    "cc_header_only_library",
)

tf_custom_op_library(
    name = "ackermann_op.so",
    srcs = ["ackermann_op.cc"],
    deps = ["//google/protobuf"],
)

py_tests(
    name = "ackermann_test",
    size = "small",
    srcs = ["ackermann_test.py"],
    data = [":ackermann_op.so"],
    tags = [
        "noasan",
        "notsan",
    ],
)

tf_custom_op_library(
    name = "look_ahead.so",
    srcs = ["look_ahead.cc"],
    gpu_srcs = ["look_ahead.cu.cc", "look_ahead.h"],
    deps = ["//google/protobuf"],
)

tf_custom_op_library(
    name = "look_ahead_grad.so",
    srcs = ["look_ahead_grad.cc"],
    gpu_srcs = ["look_ahead_grad.cu.cc", "look_ahead_grad.h"],
    deps = ["//google/protobuf"],
)

py_tests(
    name = "look_ahead_test",
    size = "small",
    srcs = ["look_ahead_op_test.py"],
    data = [":look_ahead.so"],
    tags = [
        "noasan",
        "notsan",
    ],
)

py_tests(
    name = "look_ahead_grad_test",
    size = "small",
    srcs = ["look_ahead_grad_op_test.py"],
    data = [":look_ahead_grad.so"],
    tags = [
        "noasan",
        "notsan",
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
