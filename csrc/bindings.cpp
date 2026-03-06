#include <torch/extension.h>
#include <torch/autograd.h>
#include "ops/elementwise.h"
#include "ops/reduce.h"
#include "ops/sgemm.h"
#include "ops/flash_attention.h"

// ═══════════════════════════════════════════════════════════
// 算子1：vec_add
// ═══════════════════════════════════════════════════════════
class VecAddFunction : public torch::autograd::Function<VecAddFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor a, torch::Tensor b) {
        return vec_add_forward(a, b);
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx, 
        torch::autograd::tensor_list grads
    ) {
        auto grad_c = grads[0];
        // 注意：这里假设 vec_add_backward 返回的是 pair/tuple
        // 如果它返回 vector，请改为：
        // auto res = vec_add_backward(grad_c);
        // return {res[0], res[1]};
        auto [grad_a, grad_b] = vec_add_backward(grad_c);
        return {grad_a, grad_b};
    }
};

torch::Tensor vec_add(torch::Tensor a, torch::Tensor b) {
    return VecAddFunction::apply(a, b);
}

// ═══════════════════════════════════════════════════════════
// 算子2：vec_relu
// ═══════════════════════════════════════════════════════════
class VecReLUFunction : public torch::autograd::Function<VecReLUFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x) {
        ctx->save_for_backward({x});  // 保存输入用于反向
        return vec_relu_forward(x);
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // 【修改】: 移除 .unpack()，直接获取 Tensor
        auto saved = ctx->get_saved_variables();
        auto x = saved[0]; 

        auto grad_out = grad_outputs[0];
        
        // 计算 ReLU 的反向传播
        auto grad_x = grad_out * (x > 0).to(torch::kFloat32);

        return {grad_x};
    }
};

torch::Tensor vec_relu(torch::Tensor x) {
    return VecReLUFunction::apply(x);
}

// ═══════════════════════════════════════════════════════════
// 算子3：vec_sigmoid
// ═══════════════════════════════════════════════════════════
class VecSigmoidFunction : public torch::autograd::Function<VecSigmoidFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x) {
        auto y = vec_sigmoid_forward(x);
        ctx->save_for_backward({y});  // 保存输出用于反向
        return y;
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // 【修改】: 移除 .unpack()
        auto saved = ctx->get_saved_variables();
        auto y = saved[0]; // 你在 forward 中保存的是 y (输出)

        auto grad_out = grad_outputs[0];
        
        // Sigmoid 导数: grad * y * (1 - y)
        auto grad_x = grad_out * y * (1 - y);

        return {grad_x};
    }
};

torch::Tensor vec_sigmoid(torch::Tensor x) {
    return VecSigmoidFunction::apply(x);
}

// ═══════════════════════════════════════════════════════════
// vec_sum autograd
// ═══════════════════════════════════════════════════════════
class VecSumFunction : public torch::autograd::Function<VecSumFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x) {
        ctx->save_for_backward({x});
        return vec_sum_forward(x);
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // 【修改】: 移除 .unpack()
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];

        auto grad_out = grad_outputs[0];
        
        // Sum 的导数: 全为 1
        auto grad_x = torch::ones_like(x) * grad_out;

        return {grad_x};
    }
};

torch::Tensor vec_sum(torch::Tensor x) {
    return VecSumFunction::apply(x);
}

// ═══════════════════════════════════════════════════════════
// vec_softmax autograd
// ═══════════════════════════════════════════════════════════
class VecSoftmaxFunction : public torch::autograd::Function<VecSoftmaxFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x) {
        auto y = vec_softmax_forward(x);
        ctx->save_for_backward({y});
        return y;
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // 【修改】: 移除 .unpack()
        auto saved = ctx->get_saved_variables();
        auto y = saved[0]; // Softmax 的输出

        auto grad_out = grad_outputs[0];
        
        // Softmax 导数计算 (简化版示例)
        // s = y, grad_out = dL/df
        // Jacobian: diag(s) - s^T * s
        auto s = y;
        auto ds = grad_out * s;
        auto dot = (ds).sum(-1, true);
        auto grad_x = ds - s * dot;

        return {grad_x};
    }
};

torch::Tensor vec_softmax(torch::Tensor x) {
    return VecSoftmaxFunction::apply(x);
}

// ═══════════════════════════════════════════════════════════
// Python 绑定
// ═══════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add, "Vector add with autograd");
    m.def("vec_relu", &vec_relu, "Vector relu with autograd");
    m.def("vec_sigmoid", &vec_sigmoid, "Vector sigmoid with autograd");
    m.def("vec_sum", &vec_sum, "Vector sum with autograd");
    m.def("vec_softmax", &vec_softmax, "Softmax with autograd");
    m.def("sgemm", &sgemm, "SGEMM (CUDA)");
    m.def("flash_attention", &flash_attention_forward, "FlashAttention Forward (CUDA)");
}