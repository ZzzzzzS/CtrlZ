/**
 * @file URDFParserExample.cpp
 * @brief URDF 解析器使用示例
 * @date 2026-03-03
 */

#include <iostream>
#include "CtrlZ/Utils/URDFParser.hpp"

int main()
{
    // 示例 1: 从字符串加载 URDF
    std::cout << "=== Example 1: Parse URDF from string ===" << std::endl;

    const char* urdf_string = R"(
        <?xml version="1.0"?>
        <robot name="example_robot">
            <material name="blue">
                <color rgba="0 0 0.8 1"/>
            </material>
            
            <link name="base_link">
                <inertial>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <mass value="10.0"/>
                    <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
                </inertial>
                <visual>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <geometry>
                        <box size="1 1 1"/>
                    </geometry>
                    <material name="blue"/>
                </visual>
                <collision>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <geometry>
                        <box size="1 1 1"/>
                    </geometry>
                </collision>
            </link>
            
            <link name="arm_link">
                <inertial>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <mass value="5.0"/>
                    <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
                </inertial>
                <visual>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <geometry>
                        <cylinder radius="0.1" length="1.0"/>
                    </geometry>
                </visual>
            </link>
            
            <joint name="arm_joint" type="revolute">
                <parent link="base_link"/>
                <child link="arm_link"/>
                <origin xyz="0 0 1.0" rpy="0 0 0"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
                <dynamics damping="0.1" friction="0.05"/>
            </joint>
        </robot>
    )";

    z::math::URDFParserd parser;

    if (parser.LoadFromString(urdf_string))
    {
        std::cout << "URDF parsed successfully!" << std::endl;
        parser.PrintRobotInfo();

        // 获取特定连杆的变换矩阵
        std::cout << "\n=== Link Transforms ===" << std::endl;
        auto arm_transform = parser.GetLinkTransform("arm_link");
        std::cout << "Arm link translation: " << arm_transform.GetTranslation();

        // 获取关节信息
        std::cout << "\n=== Joint Information ===" << std::endl;
        auto joints = parser.GetJoints();
        for (const auto& [name, joint] : joints)
        {
            std::cout << "Joint: " << name << std::endl;
            std::cout << "  Origin xyz: " << joint.origin_xyz;
            std::cout << "  Axis: " << joint.axis;
            if (joint.limit.has_value())
            {
                std::cout << "  Limits: [" << joint.limit->lower << ", "
                    << joint.limit->upper << "]" << std::endl;
            }
        }

        // 获取可驱动关节
        std::cout << "\n=== Actuated Joints ===" << std::endl;
        auto actuated = parser.GetActuatedJointNames();
        for (const auto& name : actuated)
        {
            std::cout << "  - " << name << std::endl;
        }
    }
    else
    {
        std::cerr << "Failed to parse URDF" << std::endl;
        return 1;
    }

    // 示例 2: 从文件加载 URDF (如果文件存在)
    std::cout << "\n=== Example 2: Parse URDF from file ===" << std::endl;
    std::cout << "Usage: parser.LoadFromFile(\"path/to/robot.urdf\");" << std::endl;

    return 0;
}
