#pragma once
#include "onnxruntime_cxx_api.h"
#include <string>
#include <locale>
#include <codecvt>
#include <iostream>
#include "Utils/MathTypes.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>

extern Ort::Env& GetOrtEnv();
extern std::wstring string_to_wstring(const std::string& str);

namespace zzs
{
	template<typename Scalar>
	math::Vector<Scalar, 3> ComputeProjectedGravity(math::Vector<Scalar, 3>& EularAngle, const math::Vector<Scalar, 3>& GravityVector)
	{
		Eigen::Matrix3<Scalar> RotMat;
		RotMat = (Eigen::AngleAxis<Scalar>(EularAngle[0], Eigen::Vector3<Scalar>::UnitX())
			* Eigen::AngleAxis<Scalar>(EularAngle[1], Eigen::Vector3<Scalar>::UnitY())
			* Eigen::AngleAxis<Scalar>(EularAngle[2], Eigen::Vector3<Scalar>::UnitZ()));
		Eigen::Vector3 <Scalar> GravityVec(GravityVector[0], GravityVector[1], GravityVector[2]);
		Eigen::Vector3 <Scalar> ProjectedGravity = RotMat.transpose() * GravityVec;

		math::Vector<Scalar, 3> ProjectedGravityVec = { ProjectedGravity[0],ProjectedGravity[1],ProjectedGravity[2] };
		return ProjectedGravityVec;
	}
};



