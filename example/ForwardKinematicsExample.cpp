/**
 * @file ForwardKinematicsExample.cpp
 * @brief 正向运动学使用示例
 * @date 2026-03-03
 */

#include <iostream>
#include "CtrlZ/Utils/ForwardKinematics.hpp"

int main()
{
    // 创建一个简单的 2-DOF 机械臂 URDF
    const char* urdf_string = R"(
        <?xml version="1.0"?>
        <robot name="2dof_arm">
            <link name="base_link">
                <inertial>
                    <origin xyz="0 0 0" rpy="0 0 0"/>
                    <mass value="1.0"/>
                    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
                </inertial>
            </link>
            
            <link name="link1">
                <inertial>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <mass value="0.5"/>
                    <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
                </inertial>
                <visual>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <geometry>
                        <cylinder radius="0.05" length="1.0"/>
                    </geometry>
                </visual>
            </link>
            
            <link name="link2">
                <inertial>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <mass value="0.3"/>
                    <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/>
                </inertial>
                <visual>
                    <origin xyz="0 0 0.5" rpy="0 0 0"/>
                    <geometry>
                        <cylinder radius="0.04" length="1.0"/>
                    </geometry>
                </visual>
            </link>
            
            <joint name="joint1" type="revolute">
                <parent link="base_link"/>
                <child link="link1"/>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <axis xyz="0 1 0"/>
                <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
            </joint>
            
            <joint name="joint2" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <origin xyz="0 0 1.0" rpy="0 0 0"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="50" velocity="5"/>
            </joint>
            
            <joint name="fixed_gripper" type="fixed">
                <parent link="link2"/>
                <child link="gripper"/>
                <origin xyz="0 0 1.0" rpy="0 0 0"/>
            </joint>
            
            <link name="gripper">
                <visual>
                    <geometry>
                        <box size="0.1 0.1 0.1"/>
                    </geometry>
                </visual>
            </link>
        </robot>
    )";

    std::cout << "=== Forward Kinematics Example ===" << std::endl;

    // 1. 解析 URDF
    z::math::URDFParserd parser;
    if (!parser.LoadFromString(urdf_string))
    {
        std::cerr << "Failed to parse URDF" << std::endl;
        return 1;
    }

    std::cout << "URDF parsed successfully!" << std::endl;
    parser.PrintRobotInfo();

    // 2. 构建正向运动学模型
    z::math::ForwardKinematicsd fk(parser);

    std::cout << "\n=== FK Model Info ===" << std::endl;
    fk.PrintModelInfo();

    // 3. 设置关节角度并计算正向运动学
    std::cout << "\n=== Forward Kinematics Calculation ===" << std::endl;

    // 测试不同的关节配置
    std::vector<std::vector<double>> test_configs = {
        {0.0, 0.0},           // 零位
        {0.0, M_PI / 4},      // joint2 = 45度
        {M_PI / 4, 0.0},      // joint1 = 45度
        {M_PI / 4, M_PI / 4}, // 都是45度
        {M_PI / 2, -M_PI / 4} // joint1 = 90度, joint2 = -45度
    };

    for (size_t i = 0; i < test_configs.size(); ++i)
    {
        const auto& q = test_configs[i];
        std::cout << "\nConfiguration " << i + 1 << ": q = ["
            << q[0] << ", " << q[1] << "] rad" << std::endl;

        // 设置关节位置
        fk.SetJointPositions(q);

        // 计算正向运动学
        fk.ComputeForwardKinematics();

        // 获取各连杆位姿
        std::cout << "  Link positions:" << std::endl;

        auto base_pos = fk.GetLinkPosition("base_link");
        std::cout << "    base_link:  ["
            << base_pos[0] << ", " << base_pos[1] << ", " << base_pos[2] << "]" << std::endl;

        auto link1_pos = fk.GetLinkPosition("link1");
        std::cout << "    link1:      ["
            << link1_pos[0] << ", " << link1_pos[1] << ", " << link1_pos[2] << "]" << std::endl;

        auto link2_pos = fk.GetLinkPosition("link2");
        std::cout << "    link2:      ["
            << link2_pos[0] << ", " << link2_pos[1] << ", " << link2_pos[2] << "]" << std::endl;

        auto gripper_pos = fk.GetLinkPosition("gripper");
        std::cout << "    gripper:    ["
            << gripper_pos[0] << ", " << gripper_pos[1] << ", " << gripper_pos[2] << "]" << std::endl;

        // 获取末端执行器姿态
        auto gripper_pose = fk.GetLinkPose("gripper");
        std::cout << "  Gripper orientation (quaternion XYZW): ["
            << gripper_pose.quaternion[0] << ", "
            << gripper_pose.quaternion[1] << ", "
            << gripper_pose.quaternion[2] << ", "
            << gripper_pose.quaternion[3] << "]" << std::endl;

        // 获取 RPY 角度
        auto rpy = gripper_pose.GetRPY();
        std::cout << "  Gripper orientation (RPY): ["
            << rpy[0] << ", " << rpy[1] << ", " << rpy[2] << "] rad" << std::endl;
    }

    // 4. 使用单独设置关节的方式
    std::cout << "\n=== Setting Joints Individually ===" << std::endl;
    fk.SetJointPosition("joint1", M_PI / 6);  // 30度
    fk.SetJointPosition("joint2", M_PI / 3);  // 60度
    fk.ComputeForwardKinematics();

    auto pos = fk.GetLinkPosition("gripper");
    std::cout << "Gripper position (joint1=30°, joint2=60°): ["
        << pos[0] << ", " << pos[1] << ", " << pos[2] << "]" << std::endl;

    // 5. 测试关节限制
    std::cout << "\n=== Joint Limits Test ===" << std::endl;
    fk.SetJointPosition("joint2", 2.0);  // 超过限制 (1.57)
    fk.ComputeForwardKinematics();
    std::cout << "Set joint2 to 2.0 rad (> limit 1.57), actual: "
        << fk.GetJointPosition("joint2") << std::endl;

    // 6. 获取所有连杆位姿
    std::cout << "\n=== All Link Poses ===" << std::endl;
    auto all_poses = fk.GetAllLinkPoses();
    for (const auto& [name, pose] : all_poses)
    {
        std::cout << name << ": pos=["
            << pose.position[0] << ", "
            << pose.position[1] << ", "
            << pose.position[2] << "]" << std::endl;
    }

    std::cout << "\n=== Example completed successfully! ===" << std::endl;

    return 0;
}
