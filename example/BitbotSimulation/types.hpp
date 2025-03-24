/**
 * @file types.hpp
 * @author Zishun Zhou
 * @brief 类型定义
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
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
#include "Workers/NetCmdWorker.hpp"

#include "Workers/NN/EraxLikeInferenceWorker.hpp"
#include "Workers/NN/HumanoidGymInferenceWorker.hpp"


#include "bitbot_mujoco/device/mujoco_imu.h"
#include "bitbot_mujoco/device/mujoco_joint.h"

using DeviceImu = bitbot::MujocoImu;
using DeviceJoint = bitbot::MujocoJoint;

/************ basic definintion***********/
using RealNumber = float;
constexpr size_t JOINT_NUMBER = 8;
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
constexpr z::CTSPair<"CurrentMotorPositionRaw", MotorVec> CurrentMotorPosRawPair;
constexpr z::CTSPair<"CurrentMotorVelocityRaw", MotorVec> CurrentMotorVelRawPair;

/********* NN pair ********************/
constexpr z::CTSPair<"NetLastAction", MotorVec> NetLastActionPair;
constexpr z::CTSPair<"NetUserCommand3", Vec3> NetCommand3Pair;
constexpr z::CTSPair<"NetProjectedGravity", Vec3> NetProjectedGravityPair;
constexpr z::CTSPair<"NetScaledAction", MotorVec> NetScaledActionPair;
constexpr z::CTSPair<"NetClockVector", z::math::Vector<RealNumber, 2>> NetClockVectorPair;
constexpr z::CTSPair<"InferenceTime", RealNumber> InferenceTimePair;

// define scheduler
using SchedulerType = z::AbstractScheduler<ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, LimitTargetMotorTorquePair,
    CurrentMotorVelRawPair, CurrentMotorPosRawPair,
    NetLastActionPair, NetCommand3Pair, NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair, InferenceTimePair>;


//define workers
using MotorResetWorkerType = z::MotorResetPositionWorker<SchedulerType, RealNumber, JOINT_NUMBER>;
using ImuWorkerType = z::ImuProcessWorker<SchedulerType, DeviceImu*, RealNumber>;
using MotorWorkerType = z::MotorControlWorker<SchedulerType, DeviceJoint*, RealNumber, JOINT_NUMBER>;
using MotorPDWorkerType = z::MotorPDControlWorker<SchedulerType, RealNumber, JOINT_NUMBER>;
using LoggerWorkerType = z::AsyncLoggerWorker<SchedulerType, RealNumber, ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, LimitTargetMotorTorquePair,
    NetLastActionPair, NetCommand3Pair, NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair, InferenceTimePair>;

using CmdWorkerType = z::NetCmdWorker<SchedulerType, RealNumber, NetCommand3Pair>;
using FlexPatchWorkerType = z::SimpleCallbackWorker<SchedulerType>;


/******define actor net************/
constexpr size_t OBSERVATION_STUCK_LENGTH = 10;
constexpr size_t OBSERVATION_EXTRA_LENGTH = 5;
//using HumanoidGymInferWorkerType = z::HumanoidGymInferenceWorker<SchedulerType, RealNumber,OBSERVATION_STUCK_LENGTH, JOINT_NUMBER>;
using EraxLikeInferWorkerType = z::EraxLikeInferenceWorker<SchedulerType, RealNumber, OBSERVATION_STUCK_LENGTH, OBSERVATION_EXTRA_LENGTH, JOINT_NUMBER>;