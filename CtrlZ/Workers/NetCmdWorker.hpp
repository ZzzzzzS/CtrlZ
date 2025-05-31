/**
 * @file NetCmdWorker.hpp
 * @author zishun zhou
 * @brief 定义了一个神经网络用户命令工人类型
 * @details 该工人类型可以用于使用键盘或手柄控制机器人运动。该类型会根据配置文档中的配置来设置命令的上下限，离散步长和反向轴等。
 * @date 2025-05-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "Workers/AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "array"
#include "Utils/MathTypes.hpp"
#include "mutex"

namespace z
{
    /**
     * @brief NetCmdWorker类型是一个神经网络用户命令工人类型，用户可以通过这个工人类型来实现神经网络用户命令的功能。
     * @details NetCmdWorker类型是一个神经网络用户命令工人类型，用户可以通过这个工人类型来实现神经网络用户命令的功能。该类型
     * 可以用于使用键盘或手柄控制机器人运动。该类型会根据配置文档中的配置来设置命令的上下限，离散步长和反向轴等。
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *    "Workers":{
     *    "Commander": {
     *        "EchoCmd": true, //是否回显打印命令
     *        "UpperLimit": [1.4,0.3,0.3], //命令上限
     *        "LowerLimit": [-1.5,-0.3,-0.3], //命令下限
     *        "DiscreteCmdStep": [0.2,0.05,0.05], //离散步长(通常用于键盘控制)
     *        "ReverseJoystickAxis": [false,true,true] //反向轴(通常用于手柄控制)
     *      }
     *    }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam CmdPrecision 命令精度，用户可以通过这个参数来指定命令的精度，比如可以指定为float或者double
     * @tparam CmdArgs 一个CTSPair类型的模板参数，用户可以通过这个参数来指定输入网络的命令名称和命令向量长度
     */
    template<typename SchedulerType, typename CmdPrecision, CTSPair CmdArgs>
    class NetCmdWorker : public AbstractWorker<SchedulerType>
    {
        using ArrayGetterType = std::array<CmdPrecision, CmdArgs.dim>;
    public:
        /**
         * @brief 构造一个神经网络用户命令工人类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        NetCmdWorker(SchedulerType::Ptr scheduler, const nlohmann::json& cfg = nlohmann::json())
            :AbstractWorker<SchedulerType>(scheduler, cfg)
        {
            nlohmann::json local_cfg = cfg;
            this->EchoCmd = local_cfg["EchoCmd"].get<bool>();
            if (local_cfg["UpperLimit"].size() != CmdArgs.dim ||
                local_cfg["LowerLimit"].size() != CmdArgs.dim ||
                local_cfg["DiscreteCmdStep"].size() != CmdArgs.dim ||
                local_cfg["ReverseJoystickAxis"].size() != CmdArgs.dim
                )
            {
                throw std::runtime_error("Command vector size mismatch with the one in config file");
            }

            this->CmdUpperLimit = static_cast<decltype(this->CmdUpperLimit)>(local_cfg["UpperLimit"].get<ArrayGetterType>());
            this->CmdLowerLimit = static_cast<decltype(this->CmdLowerLimit)>(local_cfg["LowerLimit"].get<ArrayGetterType>());
            this->CmdDiscreteStep = static_cast<decltype(this->CmdDiscreteStep)>(local_cfg["DiscreteCmdStep"].get<ArrayGetterType>());
            this->CmdReverseAxis = local_cfg["ReverseJoystickAxis"].get<decltype(this->CmdReverseAxis)>();

            this->PrintSplitLine();
            std::cout << "CommanderWorker" << std::endl;
            std::cout << "Command Vector Length:" << CmdArgs.dim << std::endl;
            std::cout << "Command Upper Limit=" << this->CmdUpperLimit;
            std::cout << "Command Lower Limit=" << this->CmdLowerLimit;
            std::cout << "Command Discrete Step=" << this->CmdDiscreteStep;
            std::cout << "Command Continuous Reverse Axis=" << this->CmdReverseAxis;
            std::cout << "Command EchoCmd= " << (this->EchoCmd ? "True" : "False") << std::endl;
            this->PrintSplitLine();
        }

        /**
         * @brief 离散增加命令，增加的步长为配置文件中设置的步长
         *
         * @param idx 命令索引
         */
        void IncreaseCmd(size_t idx)
        {
            std::lock_guard<std::mutex> lock(lock__);
            z::math::Vector<CmdPrecision, CmdArgs.dim> Cmd;
            this->Scheduler->template GetData<CmdArgs.str>(Cmd);
            Cmd[idx] += this->CmdDiscreteStep[idx];
            Cmd[idx] = std::clamp(Cmd[idx], this->CmdLowerLimit[idx], this->CmdUpperLimit[idx]);
            Cmd[idx] = (std::abs(Cmd[idx]) < 0.0001) ? 0 : Cmd[idx];
            this->Scheduler->template SetData<CmdArgs.str>(Cmd);
            if (this->EchoCmd)
            {
                std::cout << "current cmd: " << Cmd;
            }
        }

        /**
         * @brief 离散减少命令，减少的步长为配置文件中设置的步长
         *
         * @param idx 命令索引
         */
        void DecreaseCmd(size_t idx)
        {
            std::lock_guard<std::mutex> lock(lock__);
            z::math::Vector<CmdPrecision, CmdArgs.dim> Cmd;
            this->Scheduler->template GetData<CmdArgs.str>(Cmd);
            Cmd[idx] -= this->CmdDiscreteStep[idx];
            Cmd[idx] = std::clamp(Cmd[idx], this->CmdLowerLimit[idx], this->CmdUpperLimit[idx]);
            Cmd[idx] = (std::abs(Cmd[idx]) < 0.0001) ? 0 : Cmd[idx];
            this->Scheduler->template SetData<CmdArgs.str>(Cmd);
            if (this->EchoCmd)
            {
                std::cout << "current cmd: " << Cmd;
            }
        }

        /**
         * @brief 连续设置命令，根据配置文件中的上下限和是否反转轴进行缩放
         *
         * @param idx 命令索引
         * @param val 命令值
         */
        void SetCmd(size_t idx, CmdPrecision val)
        {
            val = this->CmdReverseAxis[idx] ? -val : val;
            val *= (val > 0 ? std::abs(this->CmdUpperLimit[idx]) : std::abs(this->CmdLowerLimit[idx]));
            val = std::clamp(val, this->CmdLowerLimit[idx], this->CmdUpperLimit[idx]);
            this->SetCmdNoScale(idx, val);
        }

        /**
         * @brief 连续设置命令，不进行缩放
         *
         * @param idx 命令索引
         * @param val 命令值
         */
        void SetCmdNoScale(size_t idx, CmdPrecision val)
        {
            std::lock_guard<std::mutex> lock(lock__);
            z::math::Vector<CmdPrecision, CmdArgs.dim> Cmd;
            this->Scheduler->template GetData<CmdArgs.str>(Cmd);
            Cmd[idx] = val;
            this->Scheduler->template SetData<CmdArgs.str>(Cmd);
            if (this->EchoCmd)
            {
                std::cout << "current cmd: " << Cmd;
            }
        }

        void TaskRun() override
        {
        }
    private:
        z::math::Vector<CmdPrecision, CmdArgs.dim> CmdUpperLimit;
        z::math::Vector<CmdPrecision, CmdArgs.dim> CmdLowerLimit;

        z::math::Vector<CmdPrecision, CmdArgs.dim> CmdDiscreteStep;
        std::array<bool, CmdArgs.dim> CmdReverseAxis;

        bool EchoCmd = false;
        std::mutex lock__;
    };
};