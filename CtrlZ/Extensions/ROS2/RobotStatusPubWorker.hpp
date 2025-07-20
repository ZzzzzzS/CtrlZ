#pragma once
#include "iostream"

#include "RobotStatusPubNode.hpp"
#include "Ros2Launcher.h"
#include "Schedulers/AbstractScheduler.hpp"
#include "Workers/AbstractWorker.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/MathTypes.hpp"
#include "type_traits"
#include "Utils/so3.hpp"

namespace z
{
    namespace ExtRos2
    {
        template<typename SchedulerType, typename Ros2Precision, size_t JOINT_NUMBER>
        class RobotStatusPubWorker : public AbstractWorker<SchedulerType>
        {
            static_assert(std::is_arithmetic_v<Ros2Precision>, "Ros2Precision must be an arithmetic type");
        public:
            RobotStatusPubWorker(SchedulerType::Ptr scheduler, const nlohmann::json& cfg = nlohmann::json())
                : AbstractWorker<SchedulerType>(scheduler, cfg)
            {
                this->PrintSplitLine();
                this->ros2_launcher_ = Ros2Launcher::Create();
                this->node_name_ = cfg["node_name"].get<std::string>();
                std::cout << "RobotStatusPubWorker" << std::endl;
                std::cout << "Node name: " << this->node_name_ << std::endl;
                this->PrintSplitLine();
            }

            ~RobotStatusPubWorker()
            {
            }

            void TaskCreate() override
            {
                this->robot_status_pub_node_ = RobotStatusPubNode<JOINT_NUMBER>::Create(this->node_name_);
            }

            void TaskRun() override
            {
                z::math::Vector<Ros2Precision, JOINT_NUMBER> joint_positions;
                z::math::Vector<Ros2Precision, JOINT_NUMBER> joint_velocities;
                z::math::Vector<Ros2Precision, JOINT_NUMBER> joint_efforts;
                this->Scheduler->template GetData<"CurrentMotorPosition">(joint_positions);
                this->Scheduler->template GetData<"CurrentMotorVelocity">(joint_velocities);
                this->Scheduler->template GetData<"CurrentMotorTorque">(joint_efforts);

                z::math::Vector<Ros2Precision, JOINT_NUMBER> target_joint_positions;
                z::math::Vector<Ros2Precision, JOINT_NUMBER> target_joint_velocities;
                z::math::Vector<Ros2Precision, JOINT_NUMBER> target_joint_efforts;
                this->Scheduler->template GetData<"TargetMotorPosition">(target_joint_positions);
                this->Scheduler->template GetData<"TargetMotorVelocity">(target_joint_velocities);
                this->Scheduler->template GetData<"TargetMotorTorque">(target_joint_efforts);

                z::math::Vector<Ros2Precision, 3> imu_orientation;
                z::math::Vector<Ros2Precision, 3> imu_angular_velocity;
                z::math::Vector<Ros2Precision, 3> imu_linear_acceleration;
                this->Scheduler->template GetData<"AngleValue">(imu_orientation);
                this->Scheduler->template GetData<"AngleVelocityValue">(imu_angular_velocity);
                this->Scheduler->template GetData<"AccelerationValue">(imu_linear_acceleration);

                auto quat_orientation = z::math::quat_from_euler_xyz(imu_orientation);

                this->robot_status_pub_node_->PublishImu(
                    quat_orientation.template to<double>(),
                    imu_angular_velocity.template to<double>(),
                    imu_linear_acceleration.template to<double>()
                );
                this->robot_status_pub_node_->PublishJointState(
                    joint_positions.template to<double>(),
                    joint_velocities.template to<double>(),
                    joint_efforts.template to<double>()
                );
                this->robot_status_pub_node_->PublishTargetJointState(
                    target_joint_positions.template to<double>(),
                    target_joint_velocities.template to<double>(),
                    target_joint_efforts.template to<double>()
                );

                rclcpp::spin_some(this->robot_status_pub_node_);
            }

            void TaskDestroy() override
            {
                robot_status_pub_node_.reset();
            }

        private:
            Ros2Launcher::Ptr ros2_launcher_;
            RobotStatusPubNode<JOINT_NUMBER>::Ptr robot_status_pub_node_;
            std::string node_name_;
        };
    };
};