#include "iostream"
#include "Ros2Launcher.h"

namespace z
{
    namespace ExtRos2
    {
        Ros2Launcher::Ptr Ros2Launcher::Create(int argc, char** argv)
        {
            static Ptr ptr = std::make_shared<Ros2Launcher>(argc, argv);
            return ptr;
        }

        Ros2Launcher::Ros2Launcher(int argc, char** argv)
        {
            std::cout << "Initializing ROS2..." << std::endl;
            rclcpp::init(argc, argv);
        }
        Ros2Launcher::~Ros2Launcher()
        {
            std::cout << "Ros2Launcher destructor called, shutting down ROS2..." << std::endl;
            rclcpp::shutdown();
        }

    }
}