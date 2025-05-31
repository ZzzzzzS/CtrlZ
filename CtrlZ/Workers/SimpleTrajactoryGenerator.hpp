/**
 * @file SimpleTrajactoryGenerator.hpp
 * @author zishun zhou
 * @brief 定义了一个简单的轨迹生成器
 * @details 该轨迹生成器可以生成简单的正弦波形状的轨迹
 * @date 2025-05-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "Workers/AbstractWorker.hpp"
#include "Utils/MathTypes.hpp"

namespace z
{
    template<typename SchedulerType, typename Scalar, size_t JointNumber>
    class SinTrajactoryGenerator : public z::AbstractWorker<SchedulerType>
    {
    public:

        /**
         * @brief 构造一个产生Sin曲线的工人类型
         * @details 构造一个产生Sin曲线的工人类型,可用于电机测试等。
         * @details config.json配置文件示例：
         * @code {.json}
         * {
         *   "Workers":{
         *       "SinTrajactoryGenerator": {
         *       "Amplitude": [0.5,0.5,0.5], //幅值,注意与电机数量一致
         *       "Cycle": 0.5 //周期
         *       }
         *   }
         * }
         * @endcode
         *
         * @param scheduler 调度器的指针
         * @param cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
         */
        SinTrajactoryGenerator(SchedulerType::Ptr scheduler, const nlohmann::json& cfg = nlohmann::json())
            :z::AbstractWorker<SchedulerType>(scheduler), cfg__(cfg)
        {
            nlohmann::json cfg_sin = cfg__;
            nlohmann::json AmpValue = cfg_sin["Amplitude"];
            this->Cycle__ = cfg_sin["Cycle"];
            if (AmpValue.size() != JointNumber)
            {
                throw(std::runtime_error("Amplitude size is not equal!"));
            }
            for (size_t i = 0; i < JointNumber;i++)
            {
                this->Amp__[i] = AmpValue[i];
            }
        }

        /**
         * @brief 析构函数，虚函数，用于释放资源
         *
         */
        virtual ~SinTrajactoryGenerator() {}

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，这个方法会调用用户传入的回调函数
         *
         */
        virtual void TaskRun() override
        {
            //TargetMotorPosition
            size_t t = this->Scheduler->getTimeStamp();
            z::math::Vector<Scalar, JointNumber> sin_pos;
            sin_pos.apply([this, t](Scalar& v, size_t i) {
                v = this->Amp__[i] * sin(this->Cycle__ * t * 3.1415926535);
                });
            this->Scheduler->template SetData<"TargetMotorPosition">(sin_pos);
            std::cout << "target pos" << sin_pos;
        }

        /**
         * @brief 获取配置文件
         *
         * @return 配置文件
         */
        nlohmann::json& getConfig() { return this->cfg__; }

    private:
        /// @brief 配置文件
        nlohmann::json cfg__;

        z::math::Vector<Scalar, JointNumber> Amp__;
        Scalar Cycle__;
    };

    // template<typename SchedulerType, typename Scalar, size_t JointNumber>
    // class StepTrajactoryGenerator : public z::AbstractWorker<SchedulerType>
    // {
    // public:

    //     /**
    //      * @brief 构造一个产生阶跃曲线的工人类型
    //      * @details 构造一个产生阶跃曲线的工人类型,可用于电机测试等
    //      * @details config.json配置文件示例：
    //      * {
    //      *   "Workers":{
    //      *        "SinTrajactoryGenerator": {
    //      *        "Amplitude": [0.5,0.5,0.5], //幅值,注意与电机数量一致
    //      *        "Cycle": 0.5 //周期
    //      *     }
    //      *  }
    //      *
    //      * @param scheduler 调度器的指针
    //      * @param cfg 配置文件，用户可以通过配置文件来配置工人的一些参数。
    //      */
    //     StepTrajactoryGenerator(SchedulerType::Ptr scheduler, const nlohmann::json& cfg = nlohmann::json())
    //         :z::AbstractWorker<SchedulerType>(scheduler), cfg__(cfg)
    //     {
    //         nlohmann::json cfg_step = cfg__["Workers"]["SinTrajactoryGenerator"];
    //         nlohmann::json StepUpper = cfg_step["UpperBound"];
    //         nlohmann::json StepLower = cfg_step["LowerBound"];
    //         nlohmann::json StepRandom = cfg_step["RandomStep"];
    //         this->Cycle__ = cfg_step["Cycle"];
    //         for (size_t i = 0; i < JointNumber; i++)
    //         {
    //             this->UpperBound__[i] = StepUpper[i];
    //             this->LowerBound__[i] = StepLower[i];
    //             this->RandomStep__[i] = StepRandom[i];
    //         }

    //         if (StepUpper.size() != JointNumber || StepLower.size() != JointNumber || StepRandom.size() != JointNumber)
    //         {
    //             throw(std::runtime_error("Step size is not equal!"));
    //         }
    //     }

    //     /**
    //      * @brief 析构函数，虚函数，用于释放资源
    //      *
    //      */
    //     virtual ~StepTrajactoryGenerator() {}

    //     /**
    //      * @brief TaskRun方法，在每次任务队列循环中被调用，这个方法会调用用户传入的回调函数
    //      *
    //      */
    //     virtual void TaskRun() override
    //     {
    //         //TargetMotorPosition
    //         size_t t = this->Scheduler->getTimeStamp();
    //     }

    //     /**
    //      * @brief 获取配置文件
    //      *
    //      * @return 配置文件
    //      */
    //     nlohmann::json& getConfig() { return this->cfg__; }

    // private:
    //     /// @brief 配置文件
    //     nlohmann::json cfg__;

    //     z::math::Vector<Scalar, JointNumber> UpperBound__;
    //     z::math::Vector<Scalar, JointNumber> LowerBound__;
    //     std::array<Scalar, JointNumber> RandomStep__;
    //     Scalar Cycle__;
    // };
};

