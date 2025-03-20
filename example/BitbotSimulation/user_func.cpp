/**
 * @file user_func.cpp
 * @author Zishun Zhou
 * @brief
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
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
    //读取json配置文件,并初始化各个worker
    nlohmann::json cfg_root;
    {
        //NOTE: 注意将配置文件路径修改为自己的路径
        std::string path = PROJECT_ROOT_DIR + std::string("/config.json");
        std::ifstream cfg_file(path);
        cfg_root = nlohmann::json::parse(cfg_file, nullptr, true, true);
    }

    //创建调度器
    d.TaskScheduler = new SchedulerType();

    //初始化各个worker
    d.ImuWorker = new ImuWorkerType(d.TaskScheduler, bus.GetDevice<DeviceImu>(8).value(), cfg_root);
    d.MotorWorker = new MotorWorkerType(d.TaskScheduler, cfg_root, {
          bus.GetDevice<DeviceJoint>(0).value(),
          bus.GetDevice<DeviceJoint>(1).value(),
          bus.GetDevice<DeviceJoint>(2).value(),
          bus.GetDevice<DeviceJoint>(3).value(),
          bus.GetDevice<DeviceJoint>(4).value(),
          bus.GetDevice<DeviceJoint>(5).value(),
          bus.GetDevice<DeviceJoint>(6).value(),
          bus.GetDevice<DeviceJoint>(7).value() });
    d.MotorPDWorker = new MotorPDWorkerType(d.TaskScheduler, cfg_root);
    d.Logger = new LoggerWorkerType(d.TaskScheduler, cfg_root);
    d.CommanderWorker = new CmdWorkerType(d.TaskScheduler, cfg_root);

    //创建主任务列表，并添加worker
    d.TaskScheduler->CreateTaskList("MainTask", 1, true);
    d.TaskScheduler->AddWorkers("MainTask",
        {
            d.ImuWorker,
            d.MotorPDWorker,
            d.MotorWorker
        });

    //创建推理任务列表，并添加worker，设置推理任务频率
    d.NetInferWorker = new EraxLikeInferWorkerType(d.TaskScheduler, cfg_root);
    d.TaskScheduler->CreateTaskList("InferTask", cfg_root["Scheduler"]["InferTask"]["PolicyFrequency"]);
    d.TaskScheduler->AddWorker("InferTask", d.NetInferWorker);
    d.TaskScheduler->AddWorker("InferTask", d.Logger);

    //创建复位任务列表，并添加worker，设置复位任务频率为主任务频率的1/10
    d.MotorResetWorker = new MotorResetWorkerType(d.TaskScheduler, cfg_root);
    d.TaskScheduler->CreateTaskList("ResetTask", 10);
    d.TaskScheduler->AddWorker("ResetTask", d.MotorResetWorker);

    //开始调度器
    d.TaskScheduler->Start();
}

void FinishFunc(UserData& d)
{
    //删除worker
    delete d.TaskScheduler;
    delete d.ImuWorker;
    delete d.MotorWorker;
    delete d.MotorPDWorker;
    delete d.Logger;
    delete d.NetInferWorker;
    delete d.MotorResetWorker;
    delete d.CommanderWorker;
}


std::optional<bitbot::StateId> EventInitPose(bitbot::EventValue value, UserData& d)
{
    if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    {
        d.MotorResetWorker->StartReset(); //开始复位
        d.TaskScheduler->EnableTaskList("ResetTask"); //在复位任务列表中启用复位任务
        return static_cast<bitbot::StateId>(States::PF2InitPose);
    }
    return std::optional<bitbot::StateId>();
}


std::optional<bitbot::StateId> EventPolicyRun(bitbot::EventValue value, UserData& d)
{
    if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    {
        std::cout << "policy run\n";
        d.MotorResetWorker->StopReset(); //停止复位
        d.TaskScheduler->DisableTaskList("ResetTask"); //在复位任务列表中禁用复位任务

        d.TaskScheduler->EnableTaskList("InferTask"); //在推理任务列表中启用推理任务
        return static_cast<bitbot::StateId>(States::PF2PolicyRun);
    }
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

std::optional<bitbot::StateId> EventVeloXIncrease(bitbot::EventValue keyState, UserData& d)
{
    if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    { //设置x轴速度
        d.CommanderWorker->IncreaseCmd(0);
    }
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloXDecrease(bitbot::EventValue keyState, UserData& d)
{
    if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    { //设置x轴速度
        d.CommanderWorker->DecreaseCmd(0);
    }
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYIncrease(bitbot::EventValue keyState, UserData& d)
{
    if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    {
        d.CommanderWorker->IncreaseCmd(1);
    }
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYDecrease(bitbot::EventValue keyState, UserData& d)
{
    if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    {
        d.CommanderWorker->DecreaseCmd(1);
    }
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYawIncrease(bitbot::EventValue keyState, UserData& d)
{
    if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    {
        d.CommanderWorker->IncreaseCmd(2);
    }
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYawDecrease(bitbot::EventValue keyState, UserData& d)
{
    if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up))
    {
        d.CommanderWorker->DecreaseCmd(2);
    }
    return std::optional<bitbot::StateId>();
}


std::optional<bitbot::StateId> EventJoystickXChange(bitbot::EventValue keyState, UserData& d)
{
    double vel = *reinterpret_cast<double*>(&keyState);
    d.CommanderWorker->SetCmd(0, static_cast<RealNumber>(vel));
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventJoystickYChange(bitbot::EventValue keyState, UserData& d)
{
    double vel = *reinterpret_cast<double*>(&keyState);
    d.CommanderWorker->SetCmd(1, static_cast<RealNumber>(vel));
    return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventJoystickYawChange(bitbot::EventValue keyState, UserData& d)
{
    double vel = *reinterpret_cast<double*>(&keyState);
    d.CommanderWorker->SetCmd(2, static_cast<RealNumber>(vel));
    return std::optional<bitbot::StateId>();
}


void StateWaiting(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& d)
{
    //在具体状态中只需要调用调度器的SpinOnce函数即可，调度器会根据设置自动调度任务
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
}




