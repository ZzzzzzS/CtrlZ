/**
 * @file MotorResetPositionWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/MathTypes.hpp"
#include "Utils/StaticStringUtils.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

namespace z
{
    /**
     * @brief MotorResetPositionWorker 类型是一个电机复位工人类型，用户可以通过这个工人类型来实现电机的复位功能。
     * @details MotorResetPositionWorker 类型是一个电机复位工人类型，用户可以通过这个工人类型来实现电机的复位功能。
     * 通常来说可以由调度器创建一个复位任务队列，并在复位结束后销毁这个任务队列，这个队列通常不位于主任务队列中。
     * 用户也可以通过调用StartReset方法来启动复位，通过StopReset方法来停止复位。
     * 该类型需要从数据总线中读取"CurrentMotorPosition"这个数据，并将"TargetMotorPosition"这个数据写入到数据总线中。
     * 用户需要在数据总线中注册这些数据类型，以便于工人类型能够正确的读写数据。用户可以通过配置文件来配置复位的目标位置和复位的时间。
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *    "Workers": {
     *        "MotorControl": {
     *           "DefaultPosition": [0, 0, 0, 0, 0, 0] //默认位置
     *       },
     *      "ResetPosition": {
     *         "Duration": 1.0 //复位时间 1s
     *      }
     *    }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam MotorPrecision 电机数据精度，用户可以通过这个参数来指定电机数据的精度，比如可以指定为float或者double
     * @tparam JointNumber 电机数量，用户可以通过这个参数来指定电机的数量
     */
    template<typename SchedulerType, typename MotorPrecision, size_t JointNumber>
    class MotorResetPositionWorker : public AbstractWorker<SchedulerType>
    {
        /// @brief 电机数据类型
        using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
    public:
        /**
         * @brief 构造一个电机复位工人类型
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件
         */
        MotorResetPositionWorker(SchedulerType::Ptr scheduler, const nlohmann::json& Motor_cfg, const nlohmann::json& reset_cfg)
            :AbstractWorker<SchedulerType>(scheduler),
            enabled(false)
        {
            if (Motor_cfg["DefaultPosition"].size() != JointNumber)
                throw(std::runtime_error("Default Position size is not equal!"));

            for (size_t i = 0; i < JointNumber; i++)
            {
                this->DefaultPosition[i] = Motor_cfg["DefaultPosition"][i].get<MotorPrecision>();
            }

            MotorPrecision dt = scheduler->getSpinOnceTime();
            MotorPrecision Duration = reset_cfg["Duration"];
            this->DefaultResetEpoches = static_cast<size_t>(Duration / dt);

            this->PrintSplitLine();
            std::cout << "MotorResetPositionWorker" << std::endl;
            std::cout << "DefaultPosition=" << this->DefaultPosition << std::endl;
            std::cout << "reset duration=" << Duration << std::endl;
            std::cout << "DefaultResetEpoches=" << this->DefaultResetEpoches << std::endl;

            this->PrintSplitLine();
        }

        /**
         * @brief 开始复位，用户可以通过这个方法来启动复位
         *
         * @param epoches 复位的时间，单位为调度器的时间步长，默认为0，表示使用配置文件中的时间
         */
        void StartReset(size_t epoches = 0)
        {
            this->ResetEpoches = (epoches != 0) ? epoches : this->DefaultResetEpoches;
            this->ResetCnt = this->ResetEpoches + this->Scheduler->getTimeStamp();

            this->Scheduler->template GetData<"CurrentMotorPosition">(this->TargetPosition);
            MotorValVec err = this->DefaultPosition - this->TargetPosition;
            this->PositionStep = err / static_cast<MotorPrecision>(this->ResetEpoches);
            this->enabled = true;
        }

        /**
         * @brief 停止复位，用户可以通过这个方法来停止复位
         *
         */
        void StopReset()
        {
            this->enabled = false;
        }

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，用来实现电机的复位逻辑
         *
         */
        void TaskRun() override
        {
            if (this->enabled)
            {
                if (this->Scheduler->getTimeStamp() < this->ResetCnt)
                {
                    this->TargetPosition = this->DefaultPosition - this->PositionStep * (this->ResetCnt - this->Scheduler->getTimeStamp());
                    this->Scheduler->template SetData<"TargetMotorPosition">(this->TargetPosition);
                }
                else
                {
                    this->Scheduler->template SetData<"TargetMotorPosition">(this->DefaultPosition);
                    this->enabled = false;
                }
            }
        }

    private:
        size_t DefaultResetEpoches = 0;
        size_t ResetEpoches = 0;
        size_t ResetCnt = 0;

        std::atomic<bool> enabled;

        MotorValVec DefaultPosition;
        MotorValVec PositionStep;
        MotorValVec TargetPosition;
    };
};
