#pragma once
#include <torch/extension.h>

torch::Tensor vec_sum_forward(torch::Tensor x);
torch::Tensor vec_sum_backward(torch::Tensor grad_y, torch::Tensor x);

torch::Tensor vec_softmax_forward(torch::Tensor x);
torch::Tensor vec_softmax_backward(torch::Tensor grad_y, torch::Tensor y);