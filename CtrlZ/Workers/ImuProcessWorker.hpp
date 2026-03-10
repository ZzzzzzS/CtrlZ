/**
 * @file ImuProcessWorker.hpp
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
#include "../Schedulers/AbstractScheduler.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include "../Utils/ZenBuffer.hpp"
#include "../Utils/MathTypes.hpp"

#include <iostream>
#include <memory>

namespace z
{
    /**
     * @brief 默认IMU数据访问器
     * @details 提供默认的IMU数据访问函数，要求ImuType实现GetAccX, GetAccY, GetAccZ, 
     *          GetGyroX, GetGyroY, GetGyroZ, GetRoll, GetPitch, GetYaw方法
     * 
     * @par 自定义访问器示例：
     * 当IMU传感器的接口函数名与默认值不同时，可以实现自定义访问器：
     * @code {.cpp}
     * struct MyImuAccessor {
     *     static float GetAccX(MyImu* imu)  { return imu->acc_x(); }   // 自定义函数名
     *     static float GetAccY(MyImu* imu)  { return imu->acc_y(); }
     *     static float GetAccZ(MyImu* imu)  { return imu->acc_z(); }
     *     static float GetGyroX(MyImu* imu) { return imu->gyro_x(); }
     *     static float GetGyroY(MyImu* imu) { return imu->gyro_y(); }
     *     static float GetGyroZ(MyImu* imu) { return imu->gyro_z(); }
     *     static float GetRoll(MyImu* imu)  { return imu->roll(); }
     *     static float GetPitch(MyImu* imu) { return imu->pitch(); }
     *     static float GetYaw(MyImu* imu)   { return imu->yaw(); }
     * };
     * 
     * // 使用自定义访问器实例化工人类型
     * using MyImuWorker = ImuProcessWorker<Scheduler, MyImu*, float, MyImuAccessor>;
     * @endcode
     * 
     * @tparam ImuType IMU传感器类型
     * @tparam ImuPrecision IMU数据精度
     */
    template<typename ImuType, typename ImuPrecision>
    struct DefaultImuAccessor
    {
        static ImuPrecision GetAccX(ImuType* instance) { return instance->GetAccX(); }
        static ImuPrecision GetAccY(ImuType* instance) { return instance->GetAccY(); }
        static ImuPrecision GetAccZ(ImuType* instance) { return instance->GetAccZ(); }
        static ImuPrecision GetGyroX(ImuType* instance) { return instance->GetGyroX(); }
        static ImuPrecision GetGyroY(ImuType* instance) { return instance->GetGyroY(); }
        static ImuPrecision GetGyroZ(ImuType* instance) { return instance->GetGyroZ(); }
        static ImuPrecision GetRoll(ImuType* instance) { return instance->GetRoll(); }
        static ImuPrecision GetPitch(ImuType* instance) { return instance->GetPitch(); }
        static ImuPrecision GetYaw(ImuType* instance) { return instance->GetYaw(); }
    };

    /**
     * @brief ImuProcessWorker 类型是一个IMU数据处理工人类型，这个类型用于处理IMU传感器的数据，包括加速度，角速度和角度。
     * 通常来说，这个类型可以被用于主任务队列中。
     * 这个类会在TaskCycleBegin方法中获取IMU数据并对齐进行滤波和去除异常值。用户可以通过配置文件来配置滤波器的权重。
     * @details
     * 该类会要求数据总线中包含"AccelerationRaw","AngleVelocityRaw","AngleRaw"这三个数据用于存储IMU传感器的原始数据。
     * 该类会在数据总线中存储"AccelerationValue","AngleVelocityValue","AngleValue"这三个数据用于存储滤波处理后的IMU数据。
     *
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *   "Workers": {
     *      "ImuProcess": {
     *          "AccFilterWeight": [
     *            1,  //加速度滤波权重,表示一个长度为2周期，每个周期的权重都是1的滤波器(有限长冲激响应滤波器FIR)
     *            1
     *          ],
     *          "GyroFilterWeight": [
     *            1,
     *            1
     *          ],
     *          "MagFilterWeight": [
     *            1,
     *            1
     *          ]
     *      }
     *   }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam ImuType IMU传感器类型，用户可以通过这个参数来指定IMU传感器的具体类型
     * @tparam ImuPrecision IMU数据的精度，用户可以通过这个参数来指定IMU数据的精度，比如可以指定为float或者double
     * @tparam ImuAccessor IMU数据访问器类型，默认为DefaultImuAccessor，用户可以自定义访问器来指定如何获取IMU数据
     */
    template<typename SchedulerType, typename ImuType, typename ImuPrecision, 
             typename ImuAccessor = DefaultImuAccessor<ImuType, ImuPrecision>>
    class ImuProcessWorker : public AbstractWorker<SchedulerType>
    {
        ///@brief 传感器数据必须是数值类型
        static_assert(std::is_arithmetic<ImuPrecision>::value, "ImuPrecision must be a arithmetic type");

        ///@brief 传感器数据类型
        using ImuValVec = math::Vector<ImuPrecision, 3>;

    public:
        /**
         * @brief 构造一个IMU数据处理工人类型
         *
         * @param scheduler 调度器的指针
         * @param ImuInstance IMU传感器实例指针
         * @param root_cfg 配置文件
         */
        ImuProcessWorker(SchedulerType::Ptr scheduler, ImuType ImuInstancePtr, const nlohmann::json& root_cfg = nlohmann::json())
            :AbstractWorker<SchedulerType>(scheduler),
            ImuInstance(ImuInstancePtr)
        {
            nlohmann::json cfg = root_cfg;
            this->PrintSplitLine();
            std::cout << "ImuProcessWorker" << std::endl;

            try
            {
                std::vector<ImuValVec> AccFilterWeightVec;
                std::vector<ImuValVec> GyroFilterWeightVec;
                std::vector<ImuValVec> MagFilterWeightVec;
                nlohmann::json AccFilterWeight_cfg = cfg["AccFilterWeight"];
                nlohmann::json GyroFilterWeight_cfg = cfg["GyroFilterWeight"];
                nlohmann::json MagFilterWeight_cfg = cfg["MagFilterWeight"];

                for (auto&& val : AccFilterWeight_cfg)
                {
                    ImuPrecision weight = val.get<ImuPrecision>();
                    AccFilterWeightVec.emplace_back(ImuValVec::ones() * weight);
                }

                for (auto&& val : GyroFilterWeight_cfg)
                {
                    ImuPrecision weight = val.get<ImuPrecision>();
                    GyroFilterWeightVec.emplace_back(ImuValVec::ones() * weight);
                }

                for (auto&& val : MagFilterWeight_cfg)
                {
                    ImuPrecision weight = val.get<ImuPrecision>();
                    MagFilterWeightVec.emplace_back(ImuValVec::ones() * weight);
                }

                this->AccFilter = std::make_unique<WeightFilter<ImuValVec>>(AccFilterWeightVec);
                this->GyroFilter = std::make_unique<WeightFilter<ImuValVec>>(GyroFilterWeightVec);
                this->MagFilter = std::make_unique<WeightFilter<ImuValVec>>(MagFilterWeightVec);
            }
            catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                std::cerr << "Failed to get filter weight from config, use default value." << std::endl;
                this->AccFilter = std::make_unique<WeightFilter<ImuValVec>>(std::vector<ImuValVec>(1, ImuValVec::ones()));
                this->GyroFilter = std::make_unique<WeightFilter<ImuValVec>>(std::vector<ImuValVec>(1, ImuValVec::ones()));
                this->MagFilter = std::make_unique<WeightFilter<ImuValVec>>(std::vector<ImuValVec>(1, ImuValVec::ones()));
            }

            this->PrintSplitLine();
        }

        /**
         * @brief 析构函数，虚函数，用于释放资源
         *
         */
        ~ImuProcessWorker() {}

        /**
         * @brief TaskCycleBegin方法，在每次任务队列循环的开始会被调度器调用，用于获取IMU数据并进行滤波和去除异常值。
         *
         */
        void TaskCycleBegin() override
        {
            ImuValVec Acc = {
            static_cast<ImuPrecision>(ImuAccessor::GetAccX(this->ImuInstance)),
            static_cast<ImuPrecision>(ImuAccessor::GetAccY(this->ImuInstance)),
            static_cast<ImuPrecision>(ImuAccessor::GetAccZ(this->ImuInstance))
            }; //获取Acc数据
            ImuValVec LastAcc;
            this->Scheduler->template GetData<"AccelerationRaw">(LastAcc);
            Acc = RemoveNan(Acc, LastAcc); //去除nan值，用上一次的值代替
            this->Scheduler->template SetData<"AccelerationRaw">(Acc);


            ImuValVec Gyro = {
                static_cast<ImuPrecision>(ImuAccessor::GetGyroX(this->ImuInstance)),
                static_cast<ImuPrecision>(ImuAccessor::GetGyroY(this->ImuInstance)),
                static_cast<ImuPrecision>(ImuAccessor::GetGyroZ(this->ImuInstance))
            }; //获取Gyro数据
            ImuValVec LastGyro;
            this->Scheduler->template GetData<"AngleVelocityRaw">(LastGyro);
            Gyro = RemoveNan(Gyro, LastGyro);
            this->Scheduler->template SetData<"AngleVelocityRaw">(Gyro);


            ImuValVec Mag = {
                static_cast<ImuPrecision>(ImuAccessor::GetRoll(this->ImuInstance)),
                static_cast<ImuPrecision>(ImuAccessor::GetPitch(this->ImuInstance)),
                static_cast<ImuPrecision>(ImuAccessor::GetYaw(this->ImuInstance))
            }; //获取Mag数据
            ImuValVec LastMag;
            this->Scheduler->template GetData<"AngleRaw">(LastMag);
            Mag = RemoveNan(Mag, LastMag);
            this->Scheduler->template SetData<"AngleRaw">(Mag);


            this->Scheduler->template SetData<"AccelerationValue">((*AccFilter)(Acc)); //滤波
            this->Scheduler->template SetData<"AngleVelocityValue">((*GyroFilter)(Gyro));
            this->Scheduler->template SetData<"AngleValue">((*MagFilter)(Mag));
        }

        /**
         * @brief TaskRun方法默认没有实现工作逻辑，因为对IMU数据的处理通常在流水线的开始阶段。
         *
         */
        void TaskRun() override
        {
        }

    private:

        /**
         * @brief 去除nan值，用上一次的值代替
         *
         * @param vec 待处理的数据
         * @param last_value 上一次的数据
         * @return ImuValVec 处理后的数据
         */
        ImuValVec RemoveNan(ImuValVec& vec, const ImuValVec& last_value)
        {
            vec.apply([&last_value](ImuPrecision& val, size_t idx) {
                val = std::isnan(val) ? last_value[idx] : val;
                });
            return vec;
        }

    private:
        /// @brief IMU传感器实例指针
        ImuType ImuInstance;

        /// @brief 加速度滤波器
        std::unique_ptr<WeightFilter<ImuValVec>> AccFilter;

        /// @brief 角速度滤波器
        std::unique_ptr<WeightFilter<ImuValVec>> GyroFilter;

        /// @brief 角度滤波器
        std::unique_ptr<WeightFilter<ImuValVec>> MagFilter;
    };
};

