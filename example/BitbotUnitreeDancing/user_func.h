/**
 * @file user_func.h
 * @author Zishun Zhou
 * @brief
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "bitbot_mujoco/kernel/mujoco_kernel.hpp"
#include "types.hpp"

enum Events
{
    EventInitPose = 1001,
    EventPolicyRun,
    EventSystemTest,
    EventDance,
};

enum class States : bitbot::StateId
{
    StateWaiting = 1001,
    StateInitPose,
    StatePolicyRun,
    StateSystemTest,
};

struct UserData
{
    SchedulerType::Ptr TaskScheduler;
    ImuWorkerType* ImuWorker;
    AlterImuWorkerType* AlterImuWorker;
    MotorWorkerType* MotorWorker;
    LoggerWorkerType* Logger;
    BeyondMimicUnitreeInferWorkerType* NetInferWorker;
    MotorResetWorkerType* MotorResetWorker;
    ActionManagementWorkerType* ActionManagementWorker;
    //NOTE: you don't need to delete these workers, they will be deleted by the scheduler automaticlly when the scheduler is destroyed

    std::array<DeviceJoint*, JOINT_NUMBER> JointsPtr;
    DeviceImu* ImuPtr;
    DeviceImu* ImuAlterPtr;
};

using KernelType = bitbot::MujocoKernel<UserData>;
using KernelBus = bitbot::MujocoBus;


std::optional<bitbot::StateId> EventInitPoseFunc(bitbot::EventValue value, UserData& user_data);
std::optional<bitbot::StateId> EventPolicyRunFunc(bitbot::EventValue value, UserData& user_data);
std::optional<bitbot::StateId> EventSystemTestFunc(bitbot::EventValue value, UserData& user_data);
std::optional<bitbot::StateId> EventDanceFunc(bitbot::EventValue value, UserData& user_data);

void ConfigFunc(const KernelBus& bus, UserData& d);
void FinishFunc(UserData& d);

void StateWaitingFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StateInitPoseFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StatePolicyRunFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StateSystemTestFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);