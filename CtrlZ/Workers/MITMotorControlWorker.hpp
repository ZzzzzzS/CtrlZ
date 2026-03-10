/**
 * @file MotorControlWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2026
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include "AbstractWorker.hpp"
#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/ZenBuffer.hpp"
#include "Utils/MathTypes.hpp"
#include <memory>
#include <iostream>


namespace z
{
    /**
     * @brief 默认MIT电机数据访问器
     * @details 提供默认的MIT电机数据访问函数，要求JointType实现GetActualPosition, GetActualVelocity, GetActualTorque,
     *          GetMotionKp, GetMotionKd, SetMotionKp, SetMotionKd, SetTargetTorque, SetTargetPosition, SetTargetVelocity方法
     * 
     * @par 自定义访问器示例：
     * 当电机接口的函数名与默认值不同时（例如使用CAN协议包装），可以实现自定义访问器：
     * @code {.cpp}
     * struct MyMITMotorAccessor {
     *     // 读取实际状态（getter）
     *     static float GetActualPosition(MyMotor* m) { return m->GetPosition(); }
     *     static float GetActualVelocity(MyMotor* m) { return m->GetVelocity(); }
     *     static float GetActualTorque(MyMotor* m)   { return m->GetTorque(); }
     *     
     *     // 读取电机固有参数（getter）
     *     static float GetMotionKp(MyMotor* m) { return m->GetKp(); }
     *     static float GetMotionKd(MyMotor* m) { return m->GetKd(); }
     *     
     *     // 写入目标指令（setter）
     *     static void SetMotionKp(MyMotor* m, float v)       { m->SetKp(v); }
     *     static void SetMotionKd(MyMotor* m, float v)       { m->SetKd(v); }
     *     static void SetTargetTorque(MyMotor* m, float v)   { m->SetTorque(v); }
     *     static void SetTargetPosition(MyMotor* m, float v) { m->SetPosition(v); }
     *     static void SetTargetVelocity(MyMotor* m, float v) { m->SetVelocity(v); }
     * };
     * 
     * // 使用自定义访问器实例化工人类型
     * using MyMITWorker = MITMotorControlWorker<Scheduler, MyMotor*, float, 12, MyMITMotorAccessor>;
     * @endcode
     * 
     * @tparam JointType 电机类型
     * @tparam MotorPrecision 电机数据精度
     */
    template<typename JointType, typename MotorPrecision>
    struct DefaultMITMotorAccessor
    {
        static MotorPrecision GetActualPosition(JointType joint) { return joint->GetActualPosition(); }
        static MotorPrecision GetActualVelocity(JointType joint) { return joint->GetActualVelocity(); }
        static MotorPrecision GetActualTorque(JointType joint) { return joint->GetActualTorque(); }
        static MotorPrecision GetMotionKp(JointType joint) { return joint->GetMotionKp(); }
        static MotorPrecision GetMotionKd(JointType joint) { return joint->GetMotionKd(); }
        static void SetMotionKp(JointType joint, MotorPrecision value) { joint->SetMotionKp(value); }
        static void SetMotionKd(JointType joint, MotorPrecision value) { joint->SetMotionKd(value); }
        static void SetTargetTorque(JointType joint, MotorPrecision value) { joint->SetTargetTorque(value); }
        static void SetTargetPosition(JointType joint, MotorPrecision value) { joint->SetTargetPosition(value); }
        static void SetTargetVelocity(JointType joint, MotorPrecision value) { joint->SetTargetVelocity(value); }
    };

    /**
     * @brief MITMotorControlWorker类型是一个适用于支持MIT控制模式的电机控制类型，用于读取实际电机的位置、速度、电流等数据，并将控制指令写入到电机的控制接口。
     *
     * @details MotorControlWorker类型是一个电机控制类型，用于读取实际电机的位置、速度、电流等数据，并将控制指令写入到电机的控制接口。
     * 通常来说，用户可以将该类型放在主任务队列中，用于实现电机的控制逻辑。该类型会向数据总线中写入"CurrentMotorPosition",
     * "CurrentMotorVelocity","CurrentMotorTorque"这三个数据，并读取"TargetMotorPosition","TargetMotorVelocity","TargetMotorTorque"，"TargetMotorStiffness","TargetMotorDanping"
     * 这些数据并下发给电机。
     * 用户需要在数据总线中注册这些数据类型，以便于工人类型能够正确的读写数据。用户可以通过配置文件来配置电机的控制模式。
     *
     * @details config.json配置文件示例：
     * @code {.json}
     * {
     *      "Workers": {
     *          "MotorControl": {
     *              "DefaultPosition": [0,0,0,0,0,0] //默认位置，注意这里的默认位置需要和电机的数量相匹配
     *          }
     *      }
     * }
     * @endcode
     *
     * @tparam SchedulerType 调度器类型
     * @tparam JointType 电机类型指针
     * @tparam MotorPrecision 电机数据精度，用户可以通过这个参数来指定电机数据的精度，比如可以指定为float或者double
     * @tparam JointNumber 关节电机数量
     * @tparam MotorAccessor 电机数据访问器类型，默认为DefaultMITMotorAccessor，用户可以自定义访问器来指定如何访问电机数据
     */
    template<typename SchedulerType, typename JointType, typename MotorPrecision, size_t JointNumber,
             typename MotorAccessor = DefaultMITMotorAccessor<JointType, MotorPrecision>>
    class MITMotorControlWorker : public AbstractWorker<SchedulerType>
    {
        /// @brief 电机数据类型
        using MotorValVec = math::Vector<MotorPrecision, JointNumber>;
    public:
        /**
         * @brief 构造一个电机控制工人类型
         *
         * @param scheduler 调度器的指针
         * @param root_cfg 配置文件
         * @param Joints 电机指针数组
         */
        MITMotorControlWorker(SchedulerType::Ptr scheduler, const nlohmann::json& root_cfg, const std::array<JointType, JointNumber>& Joints)
            :AbstractWorker<SchedulerType>(scheduler),
            Joints(Joints)
        {
            nlohmann::json cfg = root_cfg;
            this->PrintSplitLine();
            std::cout << "MotorControlWorker" << std::endl;
            std::cout << "JointNumber=" << JointNumber << std::endl;
            std::cout << "Default Position=" << cfg["DefaultPosition"] << std::endl;

            MotorValVec P_Gain, D_Gain;
            P_Gain.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetMotionKp(this->Joints[i]);
                });
            D_Gain.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetMotionKd(this->Joints[i]);
                });
            this->Scheduler->template SetData<"TargetMotorStiffness">(P_Gain);
            this->Scheduler->template SetData<"TargetMotorDamping">(D_Gain);
            std::cout << "Initial P Gain=" << P_Gain << std::endl;
            std::cout << "Initial D Gain=" << D_Gain << std::endl;


            this->PrintSplitLine();
        }

        /**
         * @brief 析构函数
         *
         */
        ~MITMotorControlWorker() {}

        void TaskCreate()
        {
            MotorValVec MotorVel;
            MotorValVec MotorPos;
            MotorValVec MotorTorque;
            MotorValVec P_Gain, D_Gain;

            MotorVel.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetActualVelocity(this->Joints[i]);
                });
            MotorPos.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetActualPosition(this->Joints[i]);
                });
            MotorTorque.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetActualTorque(this->Joints[i]);
                });

            P_Gain.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetMotionKp(this->Joints[i]);
                });
            D_Gain.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetMotionKd(this->Joints[i]);
                });


            this->Scheduler->template SetData<"CurrentMotorVelocity">(MotorVel);
            this->Scheduler->template SetData<"CurrentMotorPosition">(MotorPos);
            this->Scheduler->template SetData<"CurrentMotorTorque">(MotorTorque);
            this->Scheduler->template SetData<"TargetMotorStiffness">(P_Gain);
            this->Scheduler->template SetData<"TargetMotorDamping">(D_Gain);

            SetCurrentPositionAsTargetPosition();
        }

        /**
         * @brief 设置当前位置为目标位置
         *
         */
        void SetCurrentPositionAsTargetPosition()
        {
            MotorValVec MotorPos;
            MotorValVec TargetVel = MotorValVec::zeros();
            this->Scheduler->template GetData<"CurrentMotorPosition">(MotorPos);

            this->Scheduler->template SetData<"TargetMotorPosition">(MotorPos);
            this->Scheduler->template SetData<"TargetMotorVelocity">(TargetVel);
        }

        /**
         * @brief TaskCycleBegin方法，在每次任务队列循环开始时被调用，用来读取电机的位置、速度、电流等数据并写入到数据总线中
         *
         */
        void TaskCycleBegin() override
        {
            MotorValVec MotorVel;
            MotorValVec MotorPos;
            MotorValVec MotorTorque;

            MotorVel.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetActualVelocity(this->Joints[i]);
                });
            MotorPos.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetActualPosition(this->Joints[i]);
                });
            MotorTorque.apply([this](MotorPrecision& val, size_t i) {
                val = MotorAccessor::GetActualTorque(this->Joints[i]);
                });

            this->Scheduler->template SetData<"CurrentMotorVelocity">(MotorVel);
            this->Scheduler->template SetData<"CurrentMotorPosition">(MotorPos);
            this->Scheduler->template SetData<"CurrentMotorTorque">(MotorTorque);

        }

        /**
         * @brief TaskRun方法，在每次任务队列循环中被调用，用来实现电机的控制逻辑，默认为空实现，因为电机的控制逻辑在TaskCycleEnd中实现
         */
        void TaskRun() override
        {
        }

        /**
         * @brief TaskCycleEnd方法，在每次任务队列循环结束时被调用，用来将控制指令写入到电机的控制接口。
         * 限制力矩大小，以及根据控制模式选择不同的控制方式。
         */
        void TaskCycleEnd() override
        {
            MotorValVec MotorTorque;
            this->Scheduler->template GetData<"TargetMotorTorque">(MotorTorque);

            MotorValVec MotorPos;
            this->Scheduler->template GetData<"TargetMotorPosition">(MotorPos);

            MotorValVec MotorVel;
            this->Scheduler->template GetData<"TargetMotorVelocity">(MotorVel);

            MotorValVec MotorStiffness;
            this->Scheduler->template GetData<"TargetMotorStiffness">(MotorStiffness);

            MotorValVec MotorDamping;
            this->Scheduler->template GetData<"TargetMotorDamping">(MotorDamping);

            for (size_t i = 0; i < JointNumber; i++) //设置电机控制参数和控制指令
            {
                MotorAccessor::SetMotionKp(this->Joints[i], MotorStiffness[i]);
                MotorAccessor::SetMotionKd(this->Joints[i], MotorDamping[i]);
                MotorAccessor::SetTargetTorque(this->Joints[i], MotorTorque[i]);
                MotorAccessor::SetTargetPosition(this->Joints[i], MotorPos[i]);
                MotorAccessor::SetTargetVelocity(this->Joints[i], MotorVel[i]);
            }
        }

    private:
        /// @brief 电机指针数组
        std::array<JointType, JointNumber> Joints;
    };
};