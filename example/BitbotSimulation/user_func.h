#pragma once
#include "bitbot_mujoco/kernel/mujoco_kernel.hpp"
#include "types.hpp"

enum Events
{
    InitPose = 1001,
    PolicyRun,
    FakePowerOn,
    SystemTest,

    VeloxIncrease = 2001,
    VeloxDecrease = 2002,
    VeloyIncrease = 2003,
    VeloyDecrease = 2004,

    GamepadInitPose = 3002,
    GamepadPolicyRun = 3003,
    GamepadVeloxIncreaseDisc = 3101,
    GamepadVeloxDecreaseDisc = 3102,
    GamepadVeloyIncreaseDisc = 3103,
    GamepadVeloyDecreaseDisc = 3104
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
    LimxScheduler* TaskScheduler;
    LimxImuWorker* ImuWorker;
    LimxMotorWorker* MotorWorker;
    LimxLogWorker* Logger;
    LimxNetInferWorker* NetInferWorker;
    LimxMotorResetWorker* MotorResetWorker;
    //NOTE: REMEMBER TO DELETE THESE POINTERS IN FinishFunc
};

using KernelType = bitbot::MujocoKernel<UserData>;
using KernelBus = bitbot::MujocoBus;


std::optional<bitbot::StateId> EventInitPose(bitbot::EventValue value,
    UserData& user_data);
std::optional<bitbot::StateId> EventPolicyRun(bitbot::EventValue value,
    UserData& user_data);
std::optional<bitbot::StateId> EventFakePowerOn(bitbot::EventValue value,
    UserData& user_data);
std::optional<bitbot::StateId> EventSystemTest(bitbot::EventValue value,
    UserData& user_data);

std::optional<bitbot::StateId> EventVeloXIncrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloXDecrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloYIncrease(bitbot::EventValue keyState, UserData& d);
std::optional<bitbot::StateId> EventVeloYDecrease(bitbot::EventValue keyState, UserData& d);


void ConfigFunc(const KernelBus& bus, UserData& d);
void FinishFunc(UserData& d);

void StateWaiting(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& user_data);

void StateJointInitPose(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& user_data);

void StatePolicyRun(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& user_data);


void StateSystemTest(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& user_data);
