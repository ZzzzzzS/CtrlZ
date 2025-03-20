/**
 * @file user_func.h
 * @author Zishun Zhou
 * @brief
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "bitbot_mujoco/kernel/mujoco_kernel.hpp"
#include "types.hpp"

enum Events
{
    InitPose = 1001,
    PolicyRun,
    SystemTest,

    VeloxIncrease = 2001,
    VeloxDecrease = 2002,
    VeloyIncrease = 2003,
    VeloyDecrease = 2004,
    VeloYawIncrease = 2005,
    VeloYawDecrease = 2006,

    JoystickXChange = 3001,
    JoystickYChange = 3002,
    JoystickYawChange = 3003
};

enum class States : bitbot::StateId
{
    Waiting = 1001,
    PF2InitPose,
    PF2PolicyRun,
    PF2SystemTest,
};

struct UserData
{
    SchedulerType* TaskScheduler;
    ImuWorkerType* ImuWorker;
    MotorWorkerType* MotorWorker;
    MotorPDWorkerType* MotorPDWorker;
    LoggerWorkerType* Logger;
    EraxLikeInferWorkerType* NetInferWorker;
    MotorResetWorkerType* MotorResetWorker;
    CmdWorkerType* CommanderWorker;
    //NOTE: REMEMBER TO DELETE THESE POINTERS IN FinishFunc
};

using KernelType = bitbot::MujocoKernel<UserData>;
using KernelBus = bitbot::MujocoBus;


std::optional<bitbot::StateId> EventInitPose(bitbot::EventValue value, UserData& user_data);
std::optional<bitbot::StateId> EventPolicyRun(bitbot::EventValue value, UserData& user_data);
std::optional<bitbot::StateId> EventSystemTest(bitbot::EventValue value, UserData& user_data);

std::optional<bitbot::StateId> EventVeloXIncrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloXDecrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloYIncrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloYDecrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloYawIncrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloYawDecrease(bitbot::EventValue keyState, UserData& d);

std::optional<bitbot::StateId> EventJoystickXChange(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventJoystickYChange(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventJoystickYawChange(bitbot::EventValue keyState, UserData& d);


void ConfigFunc(const KernelBus& bus, UserData& d);
void FinishFunc(UserData& d);


void StateWaiting(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StateJointInitPose(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StatePolicyRun(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StateSystemTest(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
