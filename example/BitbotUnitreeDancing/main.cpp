/**
 * @file main.cpp
 * @author zishun zhou (zhouzishun@mail.zzshub.cn)
 * @brief
 *
 * @date 2025-03-04
 *
 * @copyright Copyright (c) 2025
 */

# include "bitbot_mujoco/kernel/mujoco_kernel.hpp"
#include "user_func.h"


int main(int argc, char const* argv[])
{
    //NOTE: 注意将配置文件路径修改为自己的路径
    std::string cfg_path = PROJECT_ROOT_DIR + std::string("/bitbot_mujoco.xml");
    KernelType kernel(cfg_path);

    kernel.RegisterConfigFunc(ConfigFunc);
    kernel.RegisterFinishFunc(FinishFunc);

    // 注册 Event
    kernel.RegisterEvent("system_test", static_cast<bitbot::EventId>(Events::EventSystemTest), &EventSystemTestFunc);
    kernel.RegisterEvent("init_pose", static_cast<bitbot::EventId>(Events::EventInitPose), &EventInitPoseFunc);
    kernel.RegisterEvent("policy_run", static_cast<bitbot::EventId>(Events::EventPolicyRun), &EventPolicyRunFunc);
    kernel.RegisterEvent("dance", static_cast<bitbot::EventId>(Events::EventDance), &EventDanceFunc);


    // 注册 State
    kernel.RegisterState("waiting", static_cast<bitbot::StateId>(States::StateWaiting),
        &StateWaitingFunc,
        { static_cast<bitbot::EventId>(Events::EventSystemTest), (Events::EventInitPose) });

    kernel.RegisterState("SystemTest", static_cast<bitbot::StateId>(States::StateSystemTest), &StateSystemTestFunc, {});

    kernel.RegisterState("init_pose",
        static_cast<bitbot::StateId>(States::StateInitPose),
        &StateInitPoseFunc,
        { (Events::EventPolicyRun) });


    kernel.RegisterState("policy_run",
        static_cast<bitbot::StateId>(States::StatePolicyRun),
        &StatePolicyRunFunc, { (Events::EventDance) });

    kernel.SetFirstState(static_cast<bitbot::StateId>(States::StateWaiting));
    kernel.Run(); // Run the kernel
    return 0;
}
