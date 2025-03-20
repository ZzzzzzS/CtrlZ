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
#include <Eigen/Dense>
#include <Eigen/Core>
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



        Eigen::Matrix3<Scalar> RotMat;
        RotMat = (Eigen::AngleAxis<Scalar>(EularAngle[2], Eigen::Vector3<Scalar>::UnitZ())
            * Eigen::AngleAxis<Scalar>(EularAngle[1], Eigen::Vector3<Scalar>::UnitY())
            * Eigen::AngleAxis<Scalar>(EularAngle[0], Eigen::Vector3<Scalar>::UnitX()));
        Eigen::Vector3 <Scalar> GravityVec(GravityVector[0], GravityVector[1], GravityVector[2]);
        Eigen::Vector3 <Scalar> ProjectedGravity = RotMat.transpose() * GravityVec;

        math::Vector<Scalar, 3> ProjectedGravityVec = { ProjectedGravity[0],ProjectedGravity[1],ProjectedGravity[2] };
        return ProjectedGravityVec;
    }
};



