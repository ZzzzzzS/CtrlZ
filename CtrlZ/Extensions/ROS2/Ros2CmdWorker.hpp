#pragma once
#include "Schedulers/AbstractScheduler.hpp"
#include "Workers/AbstractWorker.hpp"
#include "Utils/MathTypes.hpp"
#include "Ros2Launcher.h"
#include "functional"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "sensor_msgs/msg/joy_feedback.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"


namespace z
{
    namespace ExtRos2
    {
        template<typename SchedulerType, typename Ros2Precision>
        class Ros2CmdWorker : public AbstractWorker<SchedulerType>
        {
            static_assert(std::is_arithmetic_v<Ros2Precision>, "Ros2Precision must be an arithmetic type");
        public:
            using VelCmdCallback = std::function<void(typename SchedulerType::Ptr, const z::math::Vector<Ros2Precision, 3>&, const z::math::Vector<Ros2Precision, 3>&)>;
            using CmdArrayCallback = std::function<void(typename SchedulerType::Ptr, const std::vector<Ros2Precision>&)>;
        public:
            Ros2CmdWorker(SchedulerType::Ptr scheduler, const nlohmann::json& cfg = nlohmann::json())
                : AbstractWorker<SchedulerType>(scheduler, cfg)
            {
                this->PrintSplitLine();
                this->ros2_launcher_ = Ros2Launcher::Create();
                this->node_name_ = cfg["node_name"].get<std::string>();
                if (cfg.contains("cmd_vel_topic"))
                    this->cmd_vel_topic_ = cfg["cmd_vel_topic"].get<std::string>();
                if (cfg.contains("cmd_array_topic"))
                    this->cmd_array_topic_ = cfg["cmd_array_topic"].get<std::string>();

                std::cout << "Ros2CmdWorker" << std::endl;
                std::cout << "Node name: " << this->node_name_ << std::endl;
                this->PrintSplitLine();
            }

            void RegisterVelCmdCallback(const VelCmdCallback& callback)
            {
                this->vel_cmd_callback_ = callback;
            }

            void RegisterCmdArrayCallback(const CmdArrayCallback& callback)
            {
                this->cmd_array_callback_ = callback;
            }

            ~Ros2CmdWorker()
            {
            }

            void TaskCreate() override
            {
                this->node_ = std::make_shared<rclcpp::Node>(this->node_name_);
                this->cmd_vel_subscriber_ = this->node_->create_subscription<geometry_msgs::msg::Twist>(
                    this->cmd_vel_topic_, 10,
                    std::bind(&Ros2CmdWorker::onCmdVelReceived, this, std::placeholders::_1));
                this->cmd_array_subscriber_ = this->node_->create_subscription<std_msgs::msg::Float64MultiArray>(
                    this->cmd_array_topic_, 10,
                    std::bind(&Ros2CmdWorker::onCmdArrayReceived, this, std::placeholders::_1));
            }

            void TaskRun() override
            {
                rclcpp::spin_some(this->node_);
            }

            void TaskDestroy() override
            {
            }

        private:
            void onCmdVelReceived(const geometry_msgs::msg::Twist::SharedPtr msg)
            {
                if (this->vel_cmd_callback_) {
                    z::math::Vector<Ros2Precision, 3> linear = { static_cast<Ros2Precision>(msg->linear.x), static_cast<Ros2Precision>(msg->linear.y), static_cast<Ros2Precision>(msg->linear.z) };
                    z::math::Vector<Ros2Precision, 3> angular = { static_cast<Ros2Precision>(msg->angular.x), static_cast<Ros2Precision>(msg->angular.y), static_cast<Ros2Precision>(msg->angular.z) };
                    this->vel_cmd_callback_(this->Scheduler, linear, angular);
                }
            }

            void onCmdArrayReceived(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
            {
                if (this->cmd_array_callback_) {
                    std::vector<Ros2Precision> data;
                    data.reserve(msg->data.size());
                    for (const auto& value : msg->data) {
                        data.push_back(static_cast<Ros2Precision>(value));
                    }
                    this->cmd_array_callback_(this->Scheduler, data);
                }
            }

        private:
            Ros2Launcher::Ptr ros2_launcher_;
            std::string node_name_;
            std::string cmd_vel_topic_ = "/cmd_vel";
            std::string cmd_array_topic_ = "/cmd_array";
            rclcpp::Node::SharedPtr node_;

            //rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber_;
            rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber_;
            rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_array_subscriber_;
            //rclcpp::Publisher<sensor_msgs::msg::JoyFeedback>::SharedPtr joy_feedback_publisher_;

            CmdArrayCallback cmd_array_callback_ = nullptr;
            VelCmdCallback vel_cmd_callback_ = nullptr;

        };
    };
};