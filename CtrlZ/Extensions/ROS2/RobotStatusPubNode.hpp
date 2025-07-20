#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "Ros2Launcher.h"

namespace z
{
    namespace ExtRos2
    {
        template<size_t JOINT_NUMBER>
        class RobotStatusPubNode : public rclcpp::Node
        {
        public:
            using Ptr = std::shared_ptr<RobotStatusPubNode<JOINT_NUMBER>>;

            /// @brief Create a RobotStatusPubNode instance
            /// @param node_name Name of the node
            /// @param options Node options
            static Ptr Create(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
            {
                return std::make_shared<RobotStatusPubNode<JOINT_NUMBER>>(node_name, options);
            }

        public:
            RobotStatusPubNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
                :Node(node_name, options)
            {
                this->imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/base_imu_states", 10);
                this->joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
                this->target_joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/target_joint_states", 10);
            }

            ~RobotStatusPubNode() override = default;

            void PublishImu(
                const std::array<double, 4>& orientation,
                const std::array<double, 3>& angular_velocity,
                const std::array<double, 3>& linear_acceleration)
            {
                sensor_msgs::msg::Imu imu_msg;
                imu_msg.header.stamp = this->now();
                imu_msg.header.frame_id = "imu_frame";
                imu_msg.orientation.x = orientation[0];
                imu_msg.orientation.y = orientation[1];
                imu_msg.orientation.z = orientation[2];
                imu_msg.orientation.w = orientation[3];
                imu_msg.angular_velocity.x = angular_velocity[0];
                imu_msg.angular_velocity.y = angular_velocity[1];
                imu_msg.angular_velocity.z = angular_velocity[2];
                imu_msg.linear_acceleration.x = linear_acceleration[0];
                imu_msg.linear_acceleration.y = linear_acceleration[1];
                imu_msg.linear_acceleration.z = linear_acceleration[2];
                imu_pub_->publish(imu_msg);
            }

            void PublishJointState(
                const std::array<double, JOINT_NUMBER>& positions,
                const std::array<double, JOINT_NUMBER>& velocities,
                const std::array<double, JOINT_NUMBER>& efforts)
            {
                sensor_msgs::msg::JointState joint_state_msg;
                joint_state_msg.header.stamp = this->now();
                joint_state_msg.header.frame_id = "joint_state_frame";
                joint_state_msg.name.resize(JOINT_NUMBER);
                joint_state_msg.position.resize(JOINT_NUMBER);
                joint_state_msg.velocity.resize(JOINT_NUMBER);
                joint_state_msg.effort.resize(JOINT_NUMBER);

                for (size_t i = 0; i < JOINT_NUMBER; ++i)
                {
                    joint_state_msg.name[i] = "joint_" + std::to_string(i);
                    joint_state_msg.position[i] = positions[i];
                    joint_state_msg.velocity[i] = velocities[i];
                    joint_state_msg.effort[i] = efforts[i];
                }

                joint_state_pub_->publish(joint_state_msg);
            }

            void PublishTargetJointState(
                const std::array<double, JOINT_NUMBER>& target_positions,
                const std::array<double, JOINT_NUMBER>& target_velocities,
                const std::array<double, JOINT_NUMBER>& target_efforts)
            {
                sensor_msgs::msg::JointState target_joint_state_msg;
                target_joint_state_msg.header.stamp = this->now();
                target_joint_state_msg.header.frame_id = "target_joint_state_frame";
                target_joint_state_msg.name.resize(JOINT_NUMBER);
                target_joint_state_msg.position.resize(JOINT_NUMBER);
                target_joint_state_msg.velocity.resize(JOINT_NUMBER);
                target_joint_state_msg.effort.resize(JOINT_NUMBER);

                for (size_t i = 0; i < JOINT_NUMBER; ++i)
                {
                    target_joint_state_msg.name[i] = "target_joint_" + std::to_string(i);
                    target_joint_state_msg.position[i] = target_positions[i];
                    target_joint_state_msg.velocity[i] = target_velocities[i];
                    target_joint_state_msg.effort[i] = target_efforts[i];
                }

                target_joint_state_pub_->publish(target_joint_state_msg);
            }


        private:
            rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
            rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
            rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr target_joint_state_pub_;
        };
    };
};