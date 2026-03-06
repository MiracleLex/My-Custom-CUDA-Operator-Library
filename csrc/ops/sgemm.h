#pragma once
#include <torch/extension.h>

void sgemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta);