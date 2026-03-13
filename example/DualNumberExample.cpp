/**
 * @file DualNumberExample.cpp
 * @brief 对偶数类型使用示例
 * @date 2026-03-12
 *
 * @details 本示例演示如何使用对偶数类型进行前向模式自动微分。
 * 对偶数允许在计算函数值的同时计算导数，无需手动推导导数公式。
 */

#include <iostream>
#include <cmath>
#include "CtrlZ/Utils/DualNumber.hpp"

using namespace z::math;

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "    DualNumber (对偶数) 使用示例" << std::endl;
    std::cout << "========================================" << std::endl;

    // ============================================================
    // 1. 基本概念
    // ============================================================
    std::cout << "\n=== 1. 基本概念 ===" << std::endl;
    std::cout << "对偶数形式: x + ε*dx，其中 ε² = 0" << std::endl;
    std::cout << "- 实部 (primal): 函数值" << std::endl;
    std::cout << "- 切线部 (tangent): 导数值" << std::endl;

    // 创建一个对偶数: 3 + ε*1
    DualDouble x = make_variable(3.0);  // x = 3, dx/dx = 1
    std::cout << "\n创建变量 x = " << x << std::endl;

    // 创建常数: 5 + ε*0
    DualDouble c = make_constant(5.0);  // 常数的导数为0
    std::cout << "创建常数 c = " << c << std::endl;

    // ============================================================
    // 2. 基本算术运算
    // ============================================================
    std::cout << "\n=== 2. 基本算术运算 ===" << std::endl;

    // 加法: (x + dx) + (y + dy) = (x+y) + (dx+dy)
    auto sum = x + c;
    std::cout << "x + c = " << sum << std::endl;
    std::cout << "  解释: (3+5) + ε*(1+0) = 8 + ε*1" << std::endl;

    // 乘法: (x + ε*dx) * (y + ε*dy) = x*y + ε*(x*dy + y*dx)
    auto product = x * x;
    std::cout << "\nx * x = " << product << std::endl;
    std::cout << "  解释: 3*3 + ε*(3*1 + 3*1) = 9 + ε*6" << std::endl;
    std::cout << "  导数: d(x²)/dx = 2x = 6 ✓" << std::endl;

    // 除法
    auto quotient = x / c;
    std::cout << "\nx / c = " << quotient << std::endl;
    std::cout << "  解释: 3/5 + ε*((1*5 - 3*0)/25) = 0.6 + ε*0.2" << std::endl;

    // ============================================================
    // 3. 数学函数
    // ============================================================
    std::cout << "\n=== 3. 数学函数 ===" << std::endl;

    DualDouble t = make_variable(1.0);  // t = 1 + ε*1

    // 三角函数
    auto sin_t = sin(t);
    std::cout << "sin(1) = " << sin_t << std::endl;
    std::cout << "  验证: sin(1) = " << std::sin(1.0) << ", cos(1) = " << std::cos(1.0) << std::endl;
    std::cout << "  导数: d(sin x)/dx = cos x ✓" << std::endl;

    auto cos_t = cos(t);
    std::cout << "\ncos(1) = " << cos_t << std::endl;
    std::cout << "  导数: d(cos x)/dx = -sin x ✓" << std::endl;

    // 指数和对数
    auto exp_t = exp(t);
    std::cout << "\nexp(1) = " << exp_t << std::endl;
    std::cout << "  验证: exp(1) = " << std::exp(1.0) << std::endl;
    std::cout << "  导数: d(eˣ)/dx = eˣ ✓" << std::endl;

    auto log_t = log(t);
    std::cout << "\nlog(1) = " << log_t << std::endl;
    std::cout << "  导数: d(ln x)/dx = 1/x = 1 ✓" << std::endl;

    // 幂函数
    DualDouble base = make_variable(2.0);
    auto power = pow(base, 3.0);  // x³
    std::cout << "\n2³ = " << power << std::endl;
    std::cout << "  导数: d(x³)/dx = 3x² = 12 ✓" << std::endl;

    // ============================================================
    // 4. 复合函数示例
    // ============================================================
    std::cout << "\n=== 4. 复合函数示例 ===" << std::endl;

    // 示例: f(x) = sin(x² + 1)
    // 手动求导: f'(x) = cos(x² + 1) * 2x
    DualDouble x2 = make_variable(2.0);
    auto f1 = sin(x2 * x2 + make_constant(1.0));
    std::cout << "f(x) = sin(x² + 1) at x=2:" << std::endl;
    std::cout << "  f(2) = " << f1.primal << std::endl;
    std::cout << "  f'(2) = " << f1.tangent << std::endl;

    // 手动验证
    double manual_val = std::sin(2.0 * 2.0 + 1.0);
    double manual_grad = std::cos(2.0 * 2.0 + 1.0) * 2.0 * 2.0;
    std::cout << "  手动验证: f(2) = " << manual_val << ", f'(2) = " << manual_grad << std::endl;

    // 示例: f(x) = e^(-x²)
    // 手动求导: f'(x) = e^(-x²) * (-2x)
    DualDouble x3 = make_variable(1.0);
    auto f2 = exp(-(x3 * x3));
    std::cout << "\nf(x) = exp(-x²) at x=1:" << std::endl;
    std::cout << "  f(1) = " << f2.primal << std::endl;
    std::cout << "  f'(1) = " << f2.tangent << std::endl;

    // ============================================================
    // 5. 使用工具函数
    // ============================================================
    std::cout << "\n=== 5. 使用工具函数 ===" << std::endl;

    // value_and_grad: 同时获取函数值和导数
    std::cout << "\n--- value_and_grad ---" << std::endl;
    auto [val, grad] = value_and_grad([](auto x) {
        return x * x * x;  // f(x) = x³
        }, 2.0);
    std::cout << "f(x) = x³ at x=2:" << std::endl;
    std::cout << "  value = " << val << " (expected: 8)" << std::endl;
    std::cout << "  grad = " << grad << " (expected: 12)" << std::endl;

    // derivative: 直接获取导数
    std::cout << "\n--- derivative ---" << std::endl;
    double d = derivative([](auto x) {
        return sin(x) * exp(x);
        }, 0.0);
    std::cout << "f(x) = sin(x) * eˣ at x=0:" << std::endl;
    std::cout << "  f'(0) = " << d << " (expected: 1)" << std::endl;

    // ============================================================
    // 6. 激活函数示例
    // ============================================================
    std::cout << "\n=== 6. 激活函数示例 ===" << std::endl;

    // ReLU
    DualDouble relu_input1 = make_variable(-1.0);
    DualDouble relu_input2 = make_variable(2.0);
    std::cout << "ReLU(-1) = " << relu(relu_input1) << std::endl;
    std::cout << "ReLU(2) = " << relu(relu_input2) << std::endl;

    // Sigmoid
    DualDouble sig_input = make_variable(0.0);
    auto sig_result = sigmoid(sig_input);
    std::cout << "\nSigmoid(0) = " << sig_result << std::endl;
    std::cout << "  验证: sigmoid(0) = 0.5" << std::endl;
    std::cout << "  导数: sigmoid'(x) = sigmoid(x)*(1-sigmoid(x)) = 0.25" << std::endl;

    // Tanh
    DualDouble tanh_input = make_variable(0.0);
    auto tanh_result = tanh(tanh_input);
    std::cout << "\nTanh(0) = " << tanh_result << std::endl;
    std::cout << "  导数: tanh'(x) = 1 - tanh²(x) = 1" << std::endl;

    // ============================================================
    // 7. 多元函数示例（手动实现）
    // ============================================================
    std::cout << "\n=== 7. 多元函数偏导数示例 ===" << std::endl;
    std::cout << "f(x,y) = x² + xy + y² at (2, 3)" << std::endl;

    // 对 x 求偏导
    DualDouble x_var = make_variable(2.0);
    DualDouble y_const = make_constant(3.0);
    auto f_xy_x = x_var * x_var + x_var * y_const + y_const * y_const;
    std::cout << "∂f/∂x = " << f_xy_x.tangent << " (expected: 2*2 + 3 = 7)" << std::endl;

    // 对 y 求偏导
    DualDouble x_const = make_constant(2.0);
    DualDouble y_var = make_variable(3.0);
    auto f_xy_y = x_const * x_const + x_const * y_var + y_var * y_var;
    std::cout << "∂f/∂y = " << f_xy_y.tangent << " (expected: 2 + 2*3 = 8)" << std::endl;

    // ============================================================
    // 8. 高阶导数示例
    // ============================================================
    std::cout << "\n=== 8. 高阶导数示例 ===" << std::endl;
    std::cout << "f(x) = x³ 的二阶导数" << std::endl;

    // 使用对偶数的对偶数来计算高阶导数
    // f(x) = x³, f'(x) = 3x², f''(x) = 6x
    // 在 x = 2 处: f(2) = 8, f'(2) = 12, f''(2) = 12
    using DualDual = DualNumber<DualDouble>;

    // 创建对偶数的对偶数: x = 2 + ε₁*1, 且 dx/dx = 1 + ε₂*0
    DualDual x_higher(DualDouble(2.0, 1.0), DualDouble(1.0, 0.0));

    // 计算 x³
    DualDual result_higher = x_higher * x_higher * x_higher;

    std::cout << "f(x) = x³ at x=2:" << std::endl;
    std::cout << "  f(2) = " << result_higher.primal.primal << " (实部的实部)" << std::endl;
    std::cout << "  f'(2) = " << result_higher.primal.tangent << " (实部的切线部)" << std::endl;
    std::cout << "  f''(2) = " << result_higher.tangent.tangent << " (切线部的切线部)" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "    示例完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
