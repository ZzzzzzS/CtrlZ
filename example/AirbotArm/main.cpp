/**
 * @file main.cpp
 * @author zishun zhou (zhouzishun@mail.zzshub.cn)
 * @brief
 * @version 0.1
 * @date 2025-03-04
 *
 * @copyright Copyright (c) 2025
 * @example 使用CtrlZ控制Airbot机械臂的例子
 * 本示例大致介绍了如何使用CtrlZ控制[Airbot机械臂](https://airbots.online/)完成抓取任务
 *
 *
 */
#include "iostream"
#include "atomic"
#include "chrono"
#include "nlohmann/json.hpp"
#include "fstream"

#include "types.hpp"
#include "airbot/airbot.hpp"

SchedulerType* TaskScheduler = nullptr;
PatchWorkerType* GetArmStateWorker = nullptr;
PatchWorkerType* SetArmStateWorker = nullptr;
NNType* ArmTrackingInferenceWorker = nullptr;
LoggerWorkerType* LoggerWorker = nullptr;
KeyboardWorkerType* KeyboardWorker = nullptr;

using ArmType = arm::Robot<6>;
ArmType* ArmInstance = nullptr;
std::array<RealNumber, 6> init_q = { 0,0,0,0,0,0 };
std::array<double, 6> Kp;
std::array<double, 6> Kd;
std::array<double, 6> FeedForward;

std::atomic<bool> IsRunning = false;

void signalExitHandler(int signum)
{
    IsRunning = false;
    std::cout << "ctrl+c captured!" << std::endl;
    std::cout << "System Will Exit" << std::endl;
}

void InitArmInstance(const nlohmann::json& cfg)
{
    for (size_t i = 0; i < 6; i++)
    {
        Kp[i] = cfg["JointParam"]["Kp"][i].get<double>();
        Kd[i] = cfg["JointParam"]["Kd"][i].get<double>();
        FeedForward[i] = 0;
    }

    std::string urdf_path = cfg["UrdfPath"].get<std::string>();
    std::string can_port = cfg["CanPort"].get<std::string>();
    std::string direction = cfg["Direction"].get<std::string>();
    double max_vel = cfg["JointVelLimit"].get<double>();
    std::string EndMode = cfg["EndMode"].get<std::string>();
    std::string big_arm_type = cfg["BigArmType"].get<std::string>();
    std::string fore_arm_type = cfg["ForeArmType"].get<std::string>();

    ArmInstance = new ArmType(urdf_path, can_port, direction, max_vel, EndMode, big_arm_type, fore_arm_type);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MotorVec CurrentMotorPos = static_cast<MotorVec>(ArmInstance->get_current_joint_q());
    MotorVec CurrentMotorVel = static_cast<MotorVec>(ArmInstance->get_current_joint_v());
    MotorVec CurrentMotorTorque = static_cast<MotorVec>(ArmInstance->get_current_joint_t());
    TaskScheduler->template SetData<"CurrentMotorPosition">(CurrentMotorPos);
    TaskScheduler->template SetData<"CurrentMotorVelocity">(CurrentMotorVel);
    TaskScheduler->template SetData<"CurrentMotorTorque">(CurrentMotorTorque);
}

void SystemInit()
{
    nlohmann::json cfg_root;
    {
        //NOTE: 注意将配置文件路径修改为自己的路径
        std::string path = PROJECT_ROOT_DIR;
        path += "/config.json";
        std::ifstream cfg_file(path);
        cfg_root = nlohmann::json::parse(cfg_file, nullptr, true, true);
    }

    //init scheduler
    TaskScheduler = new SchedulerType();
    //init arm hardware
    InitArmInstance(cfg_root["Workers"]["Arm"]);

    //创建workerss
    GetArmStateWorker = new PatchWorkerType(TaskScheduler, [](SchedulerType* SchedulerPtr) {
        //get from arm sdk
        MotorVec CurrentMotorPos = static_cast<MotorVec>(ArmInstance->get_current_joint_q());
        MotorVec CurrentMotorVel = static_cast<MotorVec>(ArmInstance->get_current_joint_v());
        MotorVec CurrentMotorTorque = static_cast<MotorVec>(ArmInstance->get_current_joint_t());
        SchedulerPtr->template SetData<"CurrentMotorPosition">(CurrentMotorPos);
        SchedulerPtr->template SetData<"CurrentMotorVelocity">(CurrentMotorVel);
        SchedulerPtr->template SetData<"CurrentMotorTorque">(CurrentMotorTorque);
        }, cfg_root);

    SetArmStateWorker = new PatchWorkerType(TaskScheduler, [](SchedulerType* SchedulerPtr) {
        MotorVec TargetMotorPos;
        SchedulerPtr->template GetData<"TargetMotorPosition">(TargetMotorPos);
        MotorVec TargetMotorVel;
        SchedulerPtr->template GetData<"TargetMotorVelocity">(TargetMotorVel);
        //set to arm sdk
        ArmInstance->set_target_joint_mit(TargetMotorPos, TargetMotorVel, Kp, Kd, FeedForward);
        }, cfg_root);


    //ArmTrackingInferenceWorker = new NNType(TaskScheduler, cfg_root);
    LoggerWorker = new LoggerWorkerType(TaskScheduler, cfg_root);

    //注册键盘输入回调函数
    KeyboardWorker = new KeyboardWorkerType(TaskScheduler, cfg_root);
    KeyboardWorker->RegisterKeyCallback('w', [](SchedulerType* SchedulerPtr) {
        std::cout << "w key pressed" << std::endl;
        });


    //创建主任务列表，并添加worker
    TaskScheduler->CreateTaskList("MainTask", 1, true);
    TaskScheduler->AddWorkers("MainTask",
        {
            GetArmStateWorker, //获取电机状态
            KeyboardWorker, //获取用户键盘输入
            //ArmTrackingInferenceWorker, //推理
            SetArmStateWorker, //设置电机状态
            LoggerWorker //记录日志
        });
    TaskScheduler->Start();
}

void SystemExit()
{
    delete TaskScheduler;
    delete GetArmStateWorker;
    delete SetArmStateWorker;
    delete ArmTrackingInferenceWorker;
    delete LoggerWorker;

    //reset arm position
    ArmInstance->set_target_joint_q(init_q, true, 0.5, true);
    delete ArmInstance;
}

void SystemRun()
{
    TaskScheduler->SpinOnce();
}

int main()
{
    std::cout << "Hello, AirbotArm!" << std::endl;
    signal(SIGINT, signalExitHandler);  //处理ctrl+c退出信号
    std::cout << "System Init..." << std::endl;
    SystemInit();
    IsRunning = true;
    std::cout << "System Running" << std::endl;
    while (IsRunning)
    {
        //get time stamp
        auto start = std::chrono::steady_clock::now();

        SystemRun(); //系统运行

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto sleep_time = std::chrono::milliseconds(10) - std::chrono::milliseconds(elapsed);
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } //补偿时间满足10ms
        else {
            std::cout << "System Run Too Slow!" << std::endl;
        }
    }

    std::cout << "System Exit" << std::endl;
    SystemExit(); //处理系统退出
    std::cout << "System Exit Done, thank you for using CtrlZ!" << std::endl;
    return 0;
}