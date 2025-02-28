#include "user_func.h"

#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <memory>
#include <thread>
#include <iostream> // std::cout
#include <nlohmann/json.hpp>
#include <fstream>
#include "types.hpp"


void ConfigFunc(const KernelBus& bus, UserData& d)
{
  nlohmann::json cfg_root;
  {
    std::ifstream cfg_file("C:/Users/ZhouZishun/Documents/Workspace/BitbotLimx/config.json");
    cfg_root = nlohmann::json::parse(cfg_file, nullptr, true, true);
  }

  d.TaskScheduler = new LimxScheduler();
  d.ImuWorker = new LimxImuWorker(d.TaskScheduler, bus.GetDevice<DeviceImu>(6).value(), cfg_root);
  d.MotorWorker = new LimxMotorWorker(d.TaskScheduler, cfg_root, {
        bus.GetDevice<DeviceJoint>(0).value(),
        bus.GetDevice<DeviceJoint>(1).value(),
        bus.GetDevice<DeviceJoint>(2).value(),
        bus.GetDevice<DeviceJoint>(3).value(),
        bus.GetDevice<DeviceJoint>(4).value(),
        bus.GetDevice<DeviceJoint>(5).value() });
  d.Logger = new LimxLogWorker(d.TaskScheduler, cfg_root);

  d.TaskScheduler->CreateTaskList("MainTask", 1, true);
  d.TaskScheduler->AddWorkers("MainTask",
    {
        d.ImuWorker,
        d.MotorWorker,
        d.Logger
    });


  d.NetInferWorker = new LimxNetInferWorker(d.TaskScheduler, cfg_root);
  d.TaskScheduler->CreateTaskList("InferTask", cfg_root["Scheduler"]["InferTask"]["PolicyFrequency"]);
  d.TaskScheduler->AddWorker("InferTask", d.NetInferWorker);

  d.MotorResetWorker = new LimxMotorResetWorker(d.TaskScheduler, cfg_root);
  d.TaskScheduler->CreateTaskList("ResetTask", 10);
  d.TaskScheduler->AddWorker("ResetTask", d.MotorResetWorker);

  d.TaskScheduler->Start();
}

void FinishFunc(UserData& d)
{
  delete d.TaskScheduler;
  delete d.ImuWorker;
  delete d.MotorWorker;
  delete d.Logger;
  delete d.NetInferWorker;
  delete d.MotorResetWorker;
}


std::optional<bitbot::StateId> EventInitPose(bitbot::EventValue value, UserData& d)
{
  if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    d.MotorResetWorker->StartReset();
    d.TaskScheduler->EnableTaskList("ResetTask");
    return static_cast<bitbot::StateId>(States::PF2InitPose);
  }
  return std::optional<bitbot::StateId>();
}


std::optional<bitbot::StateId> EventPolicyRun(bitbot::EventValue value, UserData& d)
{
  if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    std::cout << "policy run\n";
    d.MotorResetWorker->StopReset();
    d.TaskScheduler->DisableTaskList("ResetTask");

    d.TaskScheduler->EnableTaskList("InferTask");
    return static_cast<bitbot::StateId>(States::PF2PolicyRun);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventFakePowerOn(bitbot::EventValue value,
  UserData& user_data)
{
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventSystemTest(bitbot::EventValue value,
  UserData& user_data)
{
  if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    return static_cast<bitbot::StateId>(States::PF2SystemTest);
  }
  return std::optional<bitbot::StateId>();
}

// velocity control callback
#define X_VEL_STEP 0.1
#define Y_VEL_STEP 0.1
std::optional<bitbot::StateId> EventVeloXIncrease(bitbot::EventValue keyState, UserData& d)
{
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    Vec3 cmd;
    d.TaskScheduler->template GetData<"NetUserCommand3">(cmd);
    cmd[0] += X_VEL_STEP;
    std::cout << "current cmd velocity: " << cmd;
    d.TaskScheduler->template SetData<"NetUserCommand3">(cmd);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloXDecrease(bitbot::EventValue keyState, UserData& d)
{
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    Vec3 cmd;
    d.TaskScheduler->template GetData<"NetUserCommand3">(cmd);
    cmd[0] -= X_VEL_STEP;
    std::cout << "current cmd velocity: " << cmd;
    d.TaskScheduler->template SetData<"NetUserCommand3">(cmd);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYIncrease(bitbot::EventValue keyState, UserData& d)
{
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    Vec3 cmd;
    d.TaskScheduler->template GetData<"NetUserCommand3">(cmd);
    cmd[1] += Y_VEL_STEP;
    std::cout << "current cmd velocity: " << cmd;
    d.TaskScheduler->template SetData<"NetUserCommand3">(cmd);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYDecrease(bitbot::EventValue keyState, UserData& d)
{
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
  {
    Vec3 cmd;
    d.TaskScheduler->template GetData<"NetUserCommand3">(cmd);
    cmd[1] -= Y_VEL_STEP;
    std::cout << "current cmd velocity: " << cmd;
    d.TaskScheduler->template SetData<"NetUserCommand3">(cmd);
  }
  return std::optional<bitbot::StateId>();
}


void StateWaiting(const bitbot::KernelInterface& kernel,
  bitbot::ExtraData& extra_data, UserData& d)
{
  d.TaskScheduler->SpinOnce();
}

void StateSystemTest(const bitbot::KernelInterface& kernel,
  bitbot::ExtraData& extra_data, UserData& user_data)
{
}


void StatePolicyRun(const bitbot::KernelInterface& kernel,
  bitbot::ExtraData& extra_data, UserData& d)
{
  d.TaskScheduler->SpinOnce();
};


void StateJointInitPose(const bitbot::KernelInterface& kernel,
  bitbot::ExtraData& extra_data, UserData& d)
{
  d.TaskScheduler->SpinOnce();
} // zyx-231007



