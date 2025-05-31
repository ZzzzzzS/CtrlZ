/**
 * @file NetInferenceWorker.h
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "onnxruntime_cxx_api.h"
#include <string>
#include <locale>
#include <codecvt>
#include <iostream>
#include "Utils/MathTypes.hpp"
#include <type_traits>

 /**
  * @brief singleton get Ort::Env object
  */
extern Ort::Env& GetOrtEnv();

/**
 * @brief convert string to wstring, this is used for windows platform
 *
 * @param str input string
 * @return std::wstring output wstring
 */
extern std::wstring string_to_wstring(const std::string& str);

namespace z
{
    /**
     * @brief Compute Projected Gravity for XYZ Eular angle
     *
     * @tparam Scalar arithmetic type, could be double or float
     * @param EularAngle XYZ Eular angle vector
     * @param GravityVector Gravity Vector
     * @return math::Vector<Scalar, 3> Projected gravity vector
     */
    template<typename Scalar>
    math::Vector<Scalar, 3> ComputeProjectedGravity(math::Vector<Scalar, 3>& EularAngle, const math::Vector<Scalar, 3>& GravityVector = { 0,0,-1 })
    {
        static_assert(std::is_arithmetic<Scalar>::value, "Scalar must be a arithmetic type");
        math::Vector<Scalar, 4> quat = math::quat_from_euler_xyz(EularAngle);
        math::Vector<Scalar, 3> pravity_vec = math::quat_rotate_inverse(quat, GravityVector);
        return pravity_vec;
    }
};



