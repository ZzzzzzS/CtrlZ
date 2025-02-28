#pragma once
#include <array>

#include "Schedulers/AbstractScheduler.hpp"

#include "Utils/StaticStringUtils.hpp"
#include "Utils/MathTypes.hpp"

#include "Workers/AbstractWorker.hpp"
#include "Workers/AsyncLoggerWorker.hpp"
#include "Workers/ImuProcessWorker.hpp"
#include "Workers/MotorControlWorker.hpp"
#include "Workers/MotorResetPositionWorker.hpp"

#include "Workers/NN/EraxLikeInferenceWorker.hpp"
#include "Workers/NN/HumanoidGymInferenceWorker.hpp"

#include "HW_Spec.hpp"


/************ basic definintion***********/
using RealNumber = float;
constexpr size_t JOINT_NUMBER = 6;
using Vec3 = z::math::Vector<RealNumber, 3>;
using MotorVec = z::math::Vector<RealNumber, JOINT_NUMBER>;

/********** IMU Data Pair******************/
constexpr z::CTSPair<"AccelerationRaw", Vec3> ImuAccRawPair;
constexpr z::CTSPair<"AngleVelocityRaw", Vec3> ImuGyroRawPair;
constexpr z::CTSPair<"AngleRaw", Vec3> ImuMagRawPair;

constexpr z::CTSPair<"AccelerationValue", Vec3> ImuAccFilteredPair;
constexpr z::CTSPair<"AngleValue", Vec3> ImuMagFilteredPair;
constexpr z::CTSPair<"AngleVelocityValue", Vec3> ImuGyroFilteredPair;

/********** Linear Velocity Pair ***********/
constexpr z::CTSPair<"LinearVelocityValue", Vec3> LinearVelocityValuePair;

/********** Motor control Pair ************/
constexpr z::CTSPair<"TargetMotorPosition", MotorVec> TargetMotorPosPair;
constexpr z::CTSPair<"TargetMotorVelocity", MotorVec> TargetMotorVelPair;
constexpr z::CTSPair<"TargetMotorTorque", MotorVec> TargetMotorTorquePair;
constexpr z::CTSPair<"CurrentMotorPosition", MotorVec> CurrentMotorPosPair;
constexpr z::CTSPair<"CurrentMotorVelocity", MotorVec> CurrentMotorVelPair;
constexpr z::CTSPair<"CurrentMotorTorque", MotorVec> CurrentMotorTorquePair;
constexpr z::CTSPair<"LimitTargetMotorTorque", MotorVec> LimitTargetMotorTorquePair;

/********* NN pair ********************/
constexpr z::CTSPair<"NetLastAction", MotorVec> NetLastActionPair;
constexpr z::CTSPair<"NetUserCommand3", Vec3> NetCommand3Pair;
constexpr z::CTSPair<"NetProjectedGravity", Vec3> NetProjectedGravityPair;
constexpr z::CTSPair<"NetScaledAction", MotorVec> NetScaledActionPair;
constexpr z::CTSPair<"NetClockVector", z::math::Vector<RealNumber, 2>> NetClockVectorPair;
constexpr z::CTSPair<"InferenceTime", RealNumber> InferenceTimePair;

// define scheduler
using LimxScheduler = z::AbstractScheduler<ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, LimitTargetMotorTorquePair,
    NetLastActionPair, NetCommand3Pair, NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair, InferenceTimePair>;


//define workers
using LimxMotorResetWorker = z::MotorResetPositionWorker<LimxScheduler, RealNumber, JOINT_NUMBER>;
using LimxImuWorker = z::ImuProcessWorker<LimxScheduler, DeviceImu*, RealNumber>;
using LimxMotorWorker = z::MotorControlWorker<LimxScheduler, DeviceJoint*, RealNumber, JOINT_NUMBER>;
using LimxLogWorker = z::AsyncLoggerWorker<LimxScheduler, RealNumber, ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, LimitTargetMotorTorquePair,
    NetLastActionPair, NetCommand3Pair, NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair, InferenceTimePair>;

using LimxFlexPatchWorker = z::SimpleCallbackWorker<LimxScheduler>;

//using LimxNetInferWorker = z::HumanoidGymInferenceWorker<LimxScheduler, RealNumber,10, JOINT_NUMBER>;
using LimxNetInferWorker = z::EraxLikeInferenceWorker<LimxScheduler, RealNumber, 10, 5, JOINT_NUMBER>;