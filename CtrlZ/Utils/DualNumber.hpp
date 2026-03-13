/**
 * @file DualNumber.hpp
 * @author zishun zhou
 * @brief 对偶数类型，用于前向模式自动微分
 *
 * @date 2026-03-12
 *
 * @copyright Copyright (c) 2026
 *
 * @details 对偶数类型实现了前向模式自动微分的基础功能。
 * 对偶数形式为 x + ε * dx，其中 ε² = 0。
 * 通过运算符重载，可以在计算函数值的同时计算导数。
 *
 * 参考 JAX 的 jvp (Jacobian-Vector Product) 设计，提供：
 * - 基本算术运算（+、-、*、/）
 * - 常用数学函数（sin、cos、exp、log 等）
 * - 工具函数（value_and_grad、jvp 等）
 *
 * 注意：本文件仅实现对偶数类型和基本运算。
 */
#pragma once
#include <cmath>
#include <iostream>
#include <type_traits>

namespace z
{
    namespace math
    {
        /**
         * @brief 对偶数类型，用于前向模式自动微分
         *
         * @tparam T 底层数值类型（如 float、double）
         *
         * @details 对偶数形式为 x + ε * dx，其中：
         * - x 是实部（primal value）
         * - dx 是切线部（tangent value）
         * - ε 是无穷小量，满足 ε² = 0
         *
         * 对偶数运算规则：
         * - (x + ε*dx) + (y + ε*dy) = (x+y) + ε*(dx+dy)
         * - (x + ε*dx) * (y + ε*dy) = (x*y) + ε*(x*dy + y*dx)
         * - f(x + ε*dx) = f(x) + ε*f'(x)*dx
         */
         // 前置声明用于类型检测
        template<typename> class DualNumber;

        /**
         * @brief 检测类型是否为对偶数
         */
        template<typename>
        struct is_dual_number : std::false_type {};

        template<typename T>
        struct is_dual_number<DualNumber<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_dual_number_v = is_dual_number<T>::value;

        /**
         * @brief 检测类型是否为算术类型或对偶数
         */
        template<typename T>
        struct is_dual_compatible {
            static constexpr bool value = std::is_arithmetic_v<T> || is_dual_number_v<T>;
        };

        template<typename T>
        inline constexpr bool is_dual_compatible_v = is_dual_compatible<T>::value;

        template<typename T>
        class DualNumber
        {
            static_assert(is_dual_compatible_v<T>, "T must be an arithmetic type or a DualNumber");

        public:
            /// @brief 实部（函数值）
            T primal;
            /// @brief 切线部（导数值）
            T tangent;

        public:
            /**
             * @brief 默认构造函数，初始化为 0 + ε*0
             */
            constexpr DualNumber() : primal(T(0)), tangent(T(0)) {}

            /**
             * @brief 从实部构造对偶数，切线部为 0
             *
             * @param p 实部值
             */
            constexpr DualNumber(T p) : primal(p), tangent(T(0)) {}

            /**
             * @brief 构造对偶数
             *
             * @param p 实部值
             * @param t 切线部值
             */
            constexpr DualNumber(T p, T t) : primal(p), tangent(t) {}

            /**
             * @brief 拷贝构造函数
             */
            constexpr DualNumber(const DualNumber&) = default;

            /**
             * @brief 移动构造函数
             */
            constexpr DualNumber(DualNumber&&) = default;

            /**
             * @brief 拷贝赋值运算符
             */
            constexpr DualNumber& operator=(const DualNumber&) = default;

            /**
             * @brief 移动赋值运算符
             */
            constexpr DualNumber& operator=(DualNumber&&) = default;

        public: // 类型转换
            /**
             * @brief 转换为其他标量类型
             *
             * @tparam Scalar 目标类型
             * @return DualNumber<Scalar> 转换后的对偶数
             */
            template<typename Scalar>
            constexpr DualNumber<Scalar> cast() const
            {
                static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be an arithmetic type");
                return DualNumber<Scalar>(static_cast<Scalar>(primal), static_cast<Scalar>(tangent));
            }

        public: // 算术运算符
            /**
             * @brief 对偶数加法
             */
            constexpr DualNumber operator+(const DualNumber& other) const
            {
                return DualNumber(primal + other.primal, tangent + other.tangent);
            }

            /**
             * @brief 对偶数与标量加法
             */
            constexpr DualNumber operator+(T scalar) const
            {
                return DualNumber(primal + scalar, tangent);
            }

            /**
             * @brief 对偶数减法
             */
            constexpr DualNumber operator-(const DualNumber& other) const
            {
                return DualNumber(primal - other.primal, tangent - other.tangent);
            }

            /**
             * @brief 对偶数与标量减法
             */
            constexpr DualNumber operator-(T scalar) const
            {
                return DualNumber(primal - scalar, tangent);
            }

            /**
             * @brief 对偶数乘法（乘积法则）
             * (x + ε*dx) * (y + ε*dy) = x*y + ε*(x*dy + y*dx)
             */
            constexpr DualNumber operator*(const DualNumber& other) const
            {
                return DualNumber(
                    primal * other.primal,
                    primal * other.tangent + tangent * other.primal
                );
            }

            /**
             * @brief 对偶数与标量乘法
             */
            constexpr DualNumber operator*(T scalar) const
            {
                return DualNumber(primal * scalar, tangent * scalar);
            }

            /**
             * @brief 对偶数除法（商法则）
             * (x + ε*dx) / (y + ε*dy) = x/y + ε*(dx*y - x*dy) / y²
             */
            constexpr DualNumber operator/(const DualNumber& other) const
            {
                T y_sq = other.primal * other.primal;
                return DualNumber(
                    primal / other.primal,
                    (tangent * other.primal - primal * other.tangent) / y_sq
                );
            }

            /**
             * @brief 对偶数与标量除法
             */
            constexpr DualNumber operator/(T scalar) const
            {
                return DualNumber(primal / scalar, tangent / scalar);
            }

            /**
             * @brief 正号运算符
             */
            constexpr DualNumber operator+() const
            {
                return *this;
            }

            /**
             * @brief 负号运算符
             */
            constexpr DualNumber operator-() const
            {
                return DualNumber(-primal, -tangent);
            }

        public: // 复合赋值运算符
            /**
             * @brief 加法赋值
             */
            constexpr DualNumber& operator+=(const DualNumber& other)
            {
                primal += other.primal;
                tangent += other.tangent;
                return *this;
            }

            /**
             * @brief 与标量加法赋值
             */
            constexpr DualNumber& operator+=(T scalar)
            {
                primal += scalar;
                return *this;
            }

            /**
             * @brief 减法赋值
             */
            constexpr DualNumber& operator-=(const DualNumber& other)
            {
                primal -= other.primal;
                tangent -= other.tangent;
                return *this;
            }

            /**
             * @brief 与标量减法赋值
             */
            constexpr DualNumber& operator-=(T scalar)
            {
                primal -= scalar;
                return *this;
            }

            /**
             * @brief 乘法赋值
             */
            constexpr DualNumber& operator*=(const DualNumber& other)
            {
                tangent = primal * other.tangent + tangent * other.primal;
                primal *= other.primal;
                return *this;
            }

            /**
             * @brief 与标量乘法赋值
             */
            constexpr DualNumber& operator*=(T scalar)
            {
                primal *= scalar;
                tangent *= scalar;
                return *this;
            }

            /**
             * @brief 除法赋值
             */
            constexpr DualNumber& operator/=(const DualNumber& other)
            {
                T y_sq = other.primal * other.primal;
                tangent = (tangent * other.primal - primal * other.tangent) / y_sq;
                primal /= other.primal;
                return *this;
            }

            /**
             * @brief 与标量除法赋值
             */
            constexpr DualNumber& operator/=(T scalar)
            {
                primal /= scalar;
                tangent /= scalar;
                return *this;
            }

        public: // 比较运算符（仅比较实部）
            /**
             * @brief 等于比较
             */
            constexpr bool operator==(const DualNumber& other) const
            {
                return primal == other.primal && tangent == other.tangent;
            }

            /**
             * @brief 不等于比较
             */
            constexpr bool operator!=(const DualNumber& other) const
            {
                return !(*this == other);
            }

            /**
             * @brief 小于比较（仅比较实部）
             */
            constexpr bool operator<(const DualNumber& other) const
            {
                return primal < other.primal;
            }

            /**
             * @brief 大于比较（仅比较实部）
             */
            constexpr bool operator>(const DualNumber& other) const
            {
                return primal > other.primal;
            }

            /**
             * @brief 小于等于比较（仅比较实部）
             */
            constexpr bool operator<=(const DualNumber& other) const
            {
                return primal <= other.primal;
            }

            /**
             * @brief 大于等于比较（仅比较实部）
             */
            constexpr bool operator>=(const DualNumber& other) const
            {
                return primal >= other.primal;
            }

        public: // 数学函数（自动微分核心）
            /**
             * @brief 正弦函数
             * sin(x + ε*dx) = sin(x) + ε*cos(x)*dx
             */
            DualNumber sin() const
            {
                return DualNumber(std::sin(primal), std::cos(primal) * tangent);
            }

            /**
             * @brief 余弦函数
             * cos(x + ε*dx) = cos(x) - ε*sin(x)*dx
             */
            DualNumber cos() const
            {
                return DualNumber(std::cos(primal), -std::sin(primal) * tangent);
            }

            /**
             * @brief 正切函数
             * tan(x + ε*dx) = tan(x) + ε*sec²(x)*dx
             */
            DualNumber tan() const
            {
                T sec = T(1) / std::cos(primal);
                T sec_sq = sec * sec;
                return DualNumber(std::tan(primal), sec_sq * tangent);
            }

            /**
             * @brief 指数函数
             * exp(x + ε*dx) = exp(x) + ε*exp(x)*dx
             */
            DualNumber exp() const
            {
                T exp_x = std::exp(primal);
                return DualNumber(exp_x, exp_x * tangent);
            }

            /**
             * @brief 自然对数
             * log(x + ε*dx) = log(x) + ε*dx/x
             */
            DualNumber log() const
            {
                return DualNumber(std::log(primal), tangent / primal);
            }

            /**
             * @brief 以2为底的对数
             */
            DualNumber log2() const
            {
                return DualNumber(std::log2(primal), tangent / (primal * std::log(T(2))));
            }

            /**
             * @brief 常用对数（以10为底）
             */
            DualNumber log10() const
            {
                return DualNumber(std::log10(primal), tangent / (primal * std::log(T(10))));
            }

            /**
             * @brief 平方根
             * sqrt(x + ε*dx) = sqrt(x) + ε*dx/(2*sqrt(x))
             */
            DualNumber sqrt() const
            {
                T sqrt_x = std::sqrt(primal);
                return DualNumber(sqrt_x, tangent / (T(2) * sqrt_x));
            }

            /**
             * @brief 幂函数
             * pow(x + ε*dx, a) = x^a + ε*a*x^(a-1)*dx
             */
            DualNumber pow(T exponent) const
            {
                T pow_x = std::pow(primal, exponent);
                return DualNumber(pow_x, exponent * std::pow(primal, exponent - T(1)) * tangent);
            }

            /**
             * @brief 平方
             */
            DualNumber square() const
            {
                return DualNumber(primal * primal, T(2) * primal * tangent);
            }

            /**
             * @brief 双曲正弦
             */
            DualNumber sinh() const
            {
                return DualNumber(std::sinh(primal), std::cosh(primal) * tangent);
            }

            /**
             * @brief 双曲余弦
             */
            DualNumber cosh() const
            {
                return DualNumber(std::cosh(primal), std::sinh(primal) * tangent);
            }

            /**
             * @brief 双曲正切
             */
            DualNumber tanh() const
            {
                T tanh_x = std::tanh(primal);
                return DualNumber(tanh_x, (T(1) - tanh_x * tanh_x) * tangent);
            }

            /**
             * @brief 反正弦
             */
            DualNumber asin() const
            {
                return DualNumber(std::asin(primal), tangent / std::sqrt(T(1) - primal * primal));
            }

            /**
             * @brief 反余弦
             */
            DualNumber acos() const
            {
                return DualNumber(std::acos(primal), -tangent / std::sqrt(T(1) - primal * primal));
            }

            /**
             * @brief 反正切
             */
            DualNumber atan() const
            {
                return DualNumber(std::atan(primal), tangent / (T(1) + primal * primal));
            }

            /**
             * @brief 绝对值
             * |x + ε*dx| = |x| + ε*sign(x)*dx
             */
            DualNumber abs() const
            {
                T sign = (primal >= T(0)) ? T(1) : T(-1);
                return DualNumber(std::abs(primal), sign * tangent);
            }

            /**
             * @brief ReLU 激活函数
             */
            DualNumber relu() const
            {
                if (primal > T(0))
                    return DualNumber(primal, tangent);
                else
                    return DualNumber(T(0), T(0));
            }

            /**
             * @brief Sigmoid 激活函数
             */
            DualNumber sigmoid() const
            {
                T sig = T(1) / (T(1) + std::exp(-primal));
                return DualNumber(sig, sig * (T(1) - sig) * tangent);
            }

            /**
             * @brief 反正切2（atan2）
             * atan2(y + ε*dy, x + ε*dx) = atan2(y, x) + ε*(x*dy - y*dx)/(x² + y²)
             */
            DualNumber atan2(const DualNumber& other) const
            {
                // atan2(y, x) 对 y 的偏导: x/(x²+y²)
                // atan2(y, x) 对 x 的偏导: -y/(x²+y²)
                T denom = other.primal * other.primal + primal * primal;
                T val = std::atan2(primal, other.primal);
                // this = y (第一个参数), other = x (第二个参数)
                // d(atan2)/dy * dy + d(atan2)/dx * dx
                T deriv = (other.primal * tangent - primal * other.tangent) / denom;
                return DualNumber(val, deriv);
            }

            /**
             * @brief 获取实部值
             */
            constexpr T value() const
            {
                return primal;
            }

            /**
             * @brief 获取导数值（切线部）
             */
            constexpr T derivative() const
            {
                return tangent;
            }

        public: // 友元函数
            /**
             * @brief 输出流运算符
             */
            friend std::ostream& operator<<(std::ostream& os, const DualNumber& d)
            {
                os << "DualNumber(" << d.primal << " + ε*" << d.tangent << ")";
                return os;
            }

            /**
             * @brief 标量与对偶数加法（非成员函数）
             */
            friend constexpr DualNumber operator+(T scalar, const DualNumber& d)
            {
                return DualNumber(scalar + d.primal, d.tangent);
            }

            /**
             * @brief 标量与对偶数减法（非成员函数）
             */
            friend constexpr DualNumber operator-(T scalar, const DualNumber& d)
            {
                return DualNumber(scalar - d.primal, -d.tangent);
            }

            /**
             * @brief 标量与对偶数乘法（非成员函数）
             */
            friend constexpr DualNumber operator*(T scalar, const DualNumber& d)
            {
                return DualNumber(scalar * d.primal, scalar * d.tangent);
            }

            /**
             * @brief 标量与对偶数除法（非成员函数）
             */
            friend constexpr DualNumber operator/(T scalar, const DualNumber& d)
            {
                T y_sq = d.primal * d.primal;
                return DualNumber(scalar / d.primal, -scalar * d.tangent / y_sq);
            }
        };

        /************************** 对偶数工具函数 **************************/

        /**
         * @brief 创建具有指定切线值的对偶数
         *
         * @tparam T 数值类型
         * @param value 实部值
         * @param tangent 切线部值
         * @return DualNumber<T> 对偶数
         *
         * @example
         * auto x = make_dual(3.0, 1.0);  // x = 3 + ε*1
         */
        template<typename T>
        constexpr DualNumber<T> make_dual(T value, T tangent)
        {
            return DualNumber<T>(value, tangent);
        }

        /**
         * @brief 创建用于求导的对偶数（切线部为1）
         *
         * @tparam T 数值类型
         * @param value 实部值
         * @return DualNumber<T> 对偶数，切线部为1
         *
         * @example
         * auto x = make_variable(3.0);  // x = 3 + ε*1
         * auto y = x * x;  // y.primal = 9, y.tangent = 6 (即 2*3)
         */
        template<typename T>
        constexpr DualNumber<T> make_variable(T value)
        {
            return DualNumber<T>(value, T(1));
        }

        /**
         * @brief 创建常数对偶数（切线部为0）
         *
         * @tparam T 数值类型
         * @param value 实部值
         * @return DualNumber<T> 对偶数，切线部为0
         */
        template<typename T>
        constexpr DualNumber<T> make_constant(T value)
        {
            return DualNumber<T>(value, T(0));
        }

        /**
         * @brief 计算函数的导数
         *
         * @tparam Func 函数类型
         * @tparam T 数值类型
         * @param f 目标函数
         * @param x 求导点
         * @return T 导数值
         *
         * @example
         * double dy = grad([](auto x) { return x * x; }, 3.0);  // dy = 6
         */
        template<typename Func, typename T>
        T grad(Func&& f, T x)
        {
            DualNumber<T> x_dual = make_variable(x);
            DualNumber<T> y_dual = f(x_dual);
            return y_dual.tangent;
        }

        /**
         * @brief 计算函数值和导数（类似 JAX 的 value_and_grad）
         *
         * @tparam Func 函数类型
         * @tparam T 数值类型
         * @param f 目标函数
         * @param x 求导点
         * @return std::pair<T, T> (函数值, 导数值)
         *
         * @example
         * auto [y, dy] = value_and_grad([](auto x) { return x * x; }, 3.0);
         * // y = 9, dy = 6
         */
        template<typename Func, typename T>
        std::pair<T, T> value_and_grad(Func&& f, T x)
        {
            DualNumber<T> x_dual = make_variable(x);
            DualNumber<T> y_dual = f(x_dual);
            return { y_dual.primal, y_dual.tangent };
        }


        /**
         * @brief 计算导数（数值微分）
         *
         * @tparam Func 函数类型
         * @tparam T 数值类型
         * @param f 目标函数
         * @param x 求导点
         * @return T 导数值
         *
         * @example
         * double dy = derivative([](auto x) { return x * x; }, 3.0);  // dy = 6
         */
        template<typename Func, typename T>
        T derivative(Func&& f, T x)
        {
            DualNumber<T> x_dual = make_variable(x);
            DualNumber<T> y_dual = f(x_dual);
            return y_dual.tangent;
        }

        /************************** 对偶数数学函数 **************************/

        /**
         * @brief 对偶数正弦函数
         */
        template<typename T>
        DualNumber<T> sin(const DualNumber<T>& d)
        {
            return d.sin();
        }

        /**
         * @brief 对偶数余弦函数
         */
        template<typename T>
        DualNumber<T> cos(const DualNumber<T>& d)
        {
            return d.cos();
        }

        /**
         * @brief 对偶数正切函数
         */
        template<typename T>
        DualNumber<T> tan(const DualNumber<T>& d)
        {
            return d.tan();
        }

        /**
         * @brief 对偶数指数函数
         */
        template<typename T>
        DualNumber<T> exp(const DualNumber<T>& d)
        {
            return d.exp();
        }

        /**
         * @brief 对偶数自然对数
         */
        template<typename T>
        DualNumber<T> log(const DualNumber<T>& d)
        {
            return d.log();
        }

        /**
         * @brief 对偶数以2为底的对数
         */
        template<typename T>
        DualNumber<T> log2(const DualNumber<T>& d)
        {
            return d.log2();
        }

        /**
         * @brief 对偶数常用对数
         */
        template<typename T>
        DualNumber<T> log10(const DualNumber<T>& d)
        {
            return d.log10();
        }

        /**
         * @brief 对偶数平方根
         */
        template<typename T>
        DualNumber<T> sqrt(const DualNumber<T>& d)
        {
            return d.sqrt();
        }

        /**
         * @brief 对偶数幂函数
         */
        template<typename T>
        DualNumber<T> pow(const DualNumber<T>& d, T exponent)
        {
            return d.pow(exponent);
        }

        /**
         * @brief 对偶数平方
         */
        template<typename T>
        DualNumber<T> square(const DualNumber<T>& d)
        {
            return d.square();
        }

        /**
         * @brief 对偶数双曲正弦
         */
        template<typename T>
        DualNumber<T> sinh(const DualNumber<T>& d)
        {
            return d.sinh();
        }

        /**
         * @brief 对偶数双曲余弦
         */
        template<typename T>
        DualNumber<T> cosh(const DualNumber<T>& d)
        {
            return d.cosh();
        }

        /**
         * @brief 对偶数双曲正切
         */
        template<typename T>
        DualNumber<T> tanh(const DualNumber<T>& d)
        {
            return d.tanh();
        }

        /**
         * @brief 对偶数反正弦
         */
        template<typename T>
        DualNumber<T> asin(const DualNumber<T>& d)
        {
            return d.asin();
        }

        /**
         * @brief 对偶数反余弦
         */
        template<typename T>
        DualNumber<T> acos(const DualNumber<T>& d)
        {
            return d.acos();
        }

        /**
         * @brief 对偶数反正切
         */
        template<typename T>
        DualNumber<T> atan(const DualNumber<T>& d)
        {
            return d.atan();
        }

        /**
         * @brief 对偶数绝对值
         */
        template<typename T>
        DualNumber<T> abs(const DualNumber<T>& d)
        {
            return d.abs();
        }

        /**
         * @brief 对偶数 ReLU
         */
        template<typename T>
        DualNumber<T> relu(const DualNumber<T>& d)
        {
            return d.relu();
        }

        /**
         * @brief 对偶数 Sigmoid
         */
        template<typename T>
        DualNumber<T> sigmoid(const DualNumber<T>& d)
        {
            return d.sigmoid();
        }

        /**
         * @brief 对偶数反正切2（atan2）
         */
        template<typename T>
        DualNumber<T> atan2(const DualNumber<T>& y, const DualNumber<T>& x)
        {
            return y.atan2(x);
        }

        /**
         * @brief 对偶数最大值
         * @details 梯度流向较大的那个输入
         */
        template<typename T>
        DualNumber<T> max(const DualNumber<T>& a, const DualNumber<T>& b)
        {
            if (a.primal > b.primal)
                return a;
            else
                return b;
        }

        /**
         * @brief 对偶数与标量最大值
         */
        template<typename T>
        DualNumber<T> max(const DualNumber<T>& a, T b)
        {
            if (a.primal > b)
                return a;
            else
                return DualNumber<T>(b, T(0));
        }

        /**
         * @brief 标量与对偶数最大值
         */
        template<typename T>
        DualNumber<T> max(T a, const DualNumber<T>& b)
        {
            if (a > b.primal)
                return DualNumber<T>(a, T(0));
            else
                return b;
        }

        /**
         * @brief 对偶数最小值
         * @details 梯度流向较小的那个输入
         */
        template<typename T>
        DualNumber<T> min(const DualNumber<T>& a, const DualNumber<T>& b)
        {
            if (a.primal < b.primal)
                return a;
            else
                return b;
        }

        /**
         * @brief 对偶数与标量最小值
         */
        template<typename T>
        DualNumber<T> min(const DualNumber<T>& a, T b)
        {
            if (a.primal < b)
                return a;
            else
                return DualNumber<T>(b, T(0));
        }

        /**
         * @brief 标量与对偶数最小值
         */
        template<typename T>
        DualNumber<T> min(T a, const DualNumber<T>& b)
        {
            if (a < b.primal)
                return DualNumber<T>(a, T(0));
            else
                return b;
        }

        /**
         * @brief 对偶数限制函数（clamp）
         * @details 将值限制在 [lo, hi] 范围内，梯度只在范围内流动
         */
        template<typename T>
        DualNumber<T> clamp(const DualNumber<T>& v, T lo, T hi)
        {
            if (v.primal < lo)
                return DualNumber<T>(lo, T(0));
            else if (v.primal > hi)
                return DualNumber<T>(hi, T(0));
            else
                return v;
        }

        /**
         * @brief 对偶数限制函数（clamp，对偶数边界）
         */
        template<typename T>
        DualNumber<T> clamp(const DualNumber<T>& v, const DualNumber<T>& lo, const DualNumber<T>& hi)
        {
            if (v.primal < lo.primal)
                return DualNumber<T>(lo.primal, T(0));
            else if (v.primal > hi.primal)
                return DualNumber<T>(hi.primal, T(0));
            else
                return v;
        }

        /************************** 常用类型别名 **************************/

        using DualFloat = DualNumber<float>;
        using DualDouble = DualNumber<double>;

    } // namespace math
} // namespace z

/************************** std 命名空间扩展 **************************/
/**
 * @brief 将对偶数自动微分函数扩展到 std 命名空间
 * @details 允许使用 std::sin(dual), std::exp(dual) 等标准调用方式
 */
namespace std
{
    /// @brief 对偶数正弦函数
    template<typename T>
    z::math::DualNumber<T> sin(const z::math::DualNumber<T>& d)
    {
        return d.sin();
    }

    /// @brief 对偶数余弦函数
    template<typename T>
    z::math::DualNumber<T> cos(const z::math::DualNumber<T>& d)
    {
        return d.cos();
    }

    /// @brief 对偶数正切函数
    template<typename T>
    z::math::DualNumber<T> tan(const z::math::DualNumber<T>& d)
    {
        return d.tan();
    }

    /// @brief 对偶数指数函数
    template<typename T>
    z::math::DualNumber<T> exp(const z::math::DualNumber<T>& d)
    {
        return d.exp();
    }

    /// @brief 对偶数自然对数
    template<typename T>
    z::math::DualNumber<T> log(const z::math::DualNumber<T>& d)
    {
        return d.log();
    }

    /// @brief 对偶数以2为底的对数
    template<typename T>
    z::math::DualNumber<T> log2(const z::math::DualNumber<T>& d)
    {
        return d.log2();
    }

    /// @brief 对偶数常用对数（以10为底）
    template<typename T>
    z::math::DualNumber<T> log10(const z::math::DualNumber<T>& d)
    {
        return d.log10();
    }

    /// @brief 对偶数平方根
    template<typename T>
    z::math::DualNumber<T> sqrt(const z::math::DualNumber<T>& d)
    {
        return d.sqrt();
    }

    /// @brief 对偶数幂函数
    template<typename T>
    z::math::DualNumber<T> pow(const z::math::DualNumber<T>& d, T exponent)
    {
        return d.pow(exponent);
    }

    /// @brief 对偶数平方
    template<typename T>
    z::math::DualNumber<T> square(const z::math::DualNumber<T>& d)
    {
        return d.square();
    }

    /// @brief 对偶数双曲正弦
    template<typename T>
    z::math::DualNumber<T> sinh(const z::math::DualNumber<T>& d)
    {
        return d.sinh();
    }

    /// @brief 对偶数双曲余弦
    template<typename T>
    z::math::DualNumber<T> cosh(const z::math::DualNumber<T>& d)
    {
        return d.cosh();
    }

    /// @brief 对偶数双曲正切
    template<typename T>
    z::math::DualNumber<T> tanh(const z::math::DualNumber<T>& d)
    {
        return d.tanh();
    }

    /// @brief 对偶数反正弦
    template<typename T>
    z::math::DualNumber<T> asin(const z::math::DualNumber<T>& d)
    {
        return d.asin();
    }

    /// @brief 对偶数反余弦
    template<typename T>
    z::math::DualNumber<T> acos(const z::math::DualNumber<T>& d)
    {
        return d.acos();
    }

    /// @brief 对偶数反正切
    template<typename T>
    z::math::DualNumber<T> atan(const z::math::DualNumber<T>& d)
    {
        return d.atan();
    }

    /// @brief 对偶数绝对值
    template<typename T>
    z::math::DualNumber<T> abs(const z::math::DualNumber<T>& d)
    {
        return d.abs();
    }

    /// @brief 对偶数 ReLU 激活函数
    template<typename T>
    z::math::DualNumber<T> relu(const z::math::DualNumber<T>& d)
    {
        return d.relu();
    }

    /// @brief 对偶数 Sigmoid 激活函数
    template<typename T>
    z::math::DualNumber<T> sigmoid(const z::math::DualNumber<T>& d)
    {
        return d.sigmoid();
    }

    /// @brief 对偶数反正切2（atan2）
    template<typename T>
    z::math::DualNumber<T> atan2(const z::math::DualNumber<T>& y, const z::math::DualNumber<T>& x)
    {
        return y.atan2(x);
    }

    /// @brief 对偶数最大值
    template<typename T>
    z::math::DualNumber<T> max(const z::math::DualNumber<T>& a, const z::math::DualNumber<T>& b)
    {
        return z::math::max(a, b);
    }

    /// @brief 对偶数与标量最大值
    template<typename T>
    z::math::DualNumber<T> max(const z::math::DualNumber<T>& a, T b)
    {
        return z::math::max(a, b);
    }

    /// @brief 标量与对偶数最大值
    template<typename T>
    z::math::DualNumber<T> max(T a, const z::math::DualNumber<T>& b)
    {
        return z::math::max(a, b);
    }

    /// @brief 对偶数最小值
    template<typename T>
    z::math::DualNumber<T> min(const z::math::DualNumber<T>& a, const z::math::DualNumber<T>& b)
    {
        return z::math::min(a, b);
    }

    /// @brief 对偶数与标量最小值
    template<typename T>
    z::math::DualNumber<T> min(const z::math::DualNumber<T>& a, T b)
    {
        return z::math::min(a, b);
    }

    /// @brief 标量与对偶数最小值
    template<typename T>
    z::math::DualNumber<T> min(T a, const z::math::DualNumber<T>& b)
    {
        return z::math::min(a, b);
    }

    /// @brief 对偶数限制函数（clamp）
    template<typename T>
    z::math::DualNumber<T> clamp(const z::math::DualNumber<T>& v, T lo, T hi)
    {
        return z::math::clamp(v, lo, hi);
    }

    /// @brief 对偶数限制函数（clamp，对偶数边界）
    template<typename T>
    z::math::DualNumber<T> clamp(const z::math::DualNumber<T>& v, const z::math::DualNumber<T>& lo, const z::math::DualNumber<T>& hi)
    {
        return z::math::clamp(v, lo, hi);
    }
} // namespace std
