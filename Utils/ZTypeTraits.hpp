#pragma once
#include <type_trait.hpp>
#include <array>

// namespace zzs
// {
//     ///@brief  check if the array is all arithmetic type
//     template<typename T, size_t N>
//     struct is_arithmetic_array : std::conjunction<std::is_arithmetic<T>, std::is_array<T>, std::integral_constant<bool, N != 0>> {};
// }