/**
 * @file types.hpp
 * @author Zishun Zhou
 * @brief 机械臂控制程序所需的一些类型定义
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once 
#include "Utils/StaticStringUtils.hpp"
#include "Utils/MathTypes.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Workers/AsyncLoggerWorker.hpp"
#include "Workers/AbstractWorker.hpp"
#include "Workers/KeyboardCommandWorker.hpp"
#include "ArmTrackingInferenceWorker.hpp"

 /************ basic definintion***********/
using RealNumber = double;
constexpr size_t JOINT_NUMBER = 6;
using Vec3 = z::math::Vector<RealNumber, 3>;
using Vec7 = z::math::Vector<RealNumber, 7>;
using Vec6 = z::math::Vector<RealNumber, 6>;
using MotorVec = z::math::Vector<RealNumber, JOINT_NUMBER>;

/********** Motor control Pair ************/
constexpr z::CTSPair<"TargetMotorPosition", MotorVec> TargetMotorPosPair;
constexpr z::CTSPair<"TargetMotorVelocity", MotorVec> TargetMotorVelPair;
constexpr z::CTSPair<"CurrentMotorPosition", MotorVec> CurrentMotorPosPair;
constexpr z::CTSPair<"CurrentMotorVelocity", MotorVec> CurrentMotorVelPair;
constexpr z::CTSPair<"CurrentMotorTorque", MotorVec> CurrentMotorTorquePair;

/********* NN pair ********************/
constexpr z::CTSPair<"NetLastAction", MotorVec> NetLastActionPair;
constexpr z::CTSPair<"NetUserCommand", Vec7> NetUserCommandPair;
constexpr z::CTSPair<"NetScaledAction", MotorVec> NetScaledActionPair;
constexpr z::CTSPair<"InferenceTime", RealNumber> InferenceTimePair;

// define scheduler
using SchedulerType = z::AbstractScheduler<TargetMotorPosPair, TargetMotorVelPair,
    CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    NetLastActionPair, NetUserCommandPair, NetScaledActionPair, InferenceTimePair>;

using PatchWorkerType = z::SimpleCallbackWorker<SchedulerType>;
using NNType = z::ArmTrackingInferenceWorker<SchedulerType, RealNumber, JOINT_NUMBER>;

using LoggerWorkerType = z::AsyncLoggerWorker<SchedulerType, RealNumber, TargetMotorPosPair, TargetMotorVelPair,
    CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    NetLastActionPair, NetUserCommandPair, NetScaledActionPair, InferenceTimePair>;

using KeyboardWorkerType = z::KeyboardCommandWorker<SchedulerType>;
