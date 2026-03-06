#pragma once
#include <torch/extension.h>

torch::Tensor vec_add_forward(torch::Tensor a, torch::Tensor b);
std::tuple<torch::Tensor, torch::Tensor> vec_add_backward(torch::Tensor grad_c);

torch::Tensor vec_relu_forward(torch::Tensor a);
torch::Tensor vec_relu_backward(torch::Tensor grad_out, torch::Tensor x);

torch::Tensor vec_sigmoid_forward(torch::Tensor a);
torch::Tensor vec_sigmoid_backward(torch::Tensor grad_out, torch::Tensor y);