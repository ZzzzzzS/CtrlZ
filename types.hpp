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

#include "../HW_Spec.hpp"


/************ basic definintion***********/
using RealNumber = float;
constexpr size_t JOINT_NUMBER = 6;
using Vec3 = zzs::math::Vector<RealNumber, 3>;
using MotorVec = zzs::math::Vector<RealNumber, JOINT_NUMBER>;

/********** IMU Data Pair******************/
constexpr zzs::CTSPair<"AccelerationRaw", Vec3> ImuAccRawPair;
constexpr zzs::CTSPair<"AngleVelocityRaw", Vec3> ImuGyroRawPair;
constexpr zzs::CTSPair<"AngleRaw", Vec3> ImuMagRawPair;

constexpr zzs::CTSPair<"AccelerationValue", Vec3> ImuAccFilteredPair;
constexpr zzs::CTSPair<"AngleValue", Vec3> ImuMagFilteredPair;
constexpr zzs::CTSPair<"AngleVelocityValue", Vec3> ImuGyroFilteredPair;

/********** Linear Velocity Pair ***********/
constexpr zzs::CTSPair<"LinearVelocityValue", Vec3> LinearVelocityValuePair;

/********** Motor control Pair ************/
constexpr zzs::CTSPair<"TargetMotorPosition", MotorVec> TargetMotorPosPair;
constexpr zzs::CTSPair<"TargetMotorVelocity", MotorVec> TargetMotorVelPair;
constexpr zzs::CTSPair<"TargetMotorTorque", MotorVec> TargetMotorTorquePair;
constexpr zzs::CTSPair<"CurrentMotorPosition", MotorVec> CurrentMotorPosPair;
constexpr zzs::CTSPair<"CurrentMotorVelocity", MotorVec> CurrentMotorVelPair;
constexpr zzs::CTSPair<"CurrentMotorTorque", MotorVec> CurrentMotorTorquePair;
constexpr zzs::CTSPair<"LimitTargetMotorTorque", MotorVec> LimitTargetMotorTorquePair;

/********* NN pair ********************/
constexpr zzs::CTSPair<"NetLastAction", MotorVec> NetLastActionPair;
constexpr zzs::CTSPair<"NetUserCommand3", Vec3> NetCommand3Pair;
constexpr zzs::CTSPair<"NetProjectedGravity", Vec3> NetProjectedGravityPair;
constexpr zzs::CTSPair<"NetScaledAction", MotorVec> NetScaledActionPair;
constexpr zzs::CTSPair<"NetClockVector", zzs::math::Vector<RealNumber, 2>> NetClockVectorPair;
constexpr zzs::CTSPair<"InferenceTime", RealNumber> InferenceTimePair;

// define scheduler
using LimxScheduler = zzs::AbstractScheduler<ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, LimitTargetMotorTorquePair,
    NetLastActionPair, NetCommand3Pair, NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair, InferenceTimePair>;


//define workers
using LimxMotorResetWorker = zzs::MotorResetPositionWorker<LimxScheduler, RealNumber, JOINT_NUMBER>;
using LimxImuWorker = zzs::ImuProcessWorker<LimxScheduler, DeviceImu*, RealNumber>;
using LimxMotorWorker = zzs::MotorControlWorker<LimxScheduler, DeviceJoint*, RealNumber, JOINT_NUMBER>;
using LimxLogWorker = zzs::AsyncLoggerWorker<LimxScheduler, RealNumber, ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, LimitTargetMotorTorquePair,
    NetLastActionPair, NetCommand3Pair, NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair, InferenceTimePair>;

using LimxFlexPatchWorker = zzs::SimpleCallbackWorker<LimxScheduler>;

//using LimxNetInferWorker = zzs::HumanoidGymInferenceWorker<LimxScheduler, RealNumber,10, JOINT_NUMBER>;
using LimxNetInferWorker = zzs::EraxLikeInferenceWorker<LimxScheduler, RealNumber, 10, 5, JOINT_NUMBER>;