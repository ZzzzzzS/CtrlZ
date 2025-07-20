#pragma once
#include "rclcpp/rclcpp.hpp"
#include "memory"

namespace z
{
    namespace ExtRos2
    {
        class Ros2Launcher
        {
        public:
            using Ptr = std::shared_ptr<Ros2Launcher>;

            /// @brief Create a singleton instance of Ros2Launcher
            static Ptr Create(int argc = 0, char** argv = nullptr);

        public:
            Ros2Launcher(int argc, char** argv);
            ~Ros2Launcher();
        };
    };
};