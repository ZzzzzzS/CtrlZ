/**
 * @file ForwardKinematics.hpp
 * @brief 机器人正向运动学计算工具
 * @date 2026-03-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <stack>
#include <queue>
#include <stdexcept>

#include "URDFParser.hpp"
#include "so3.hpp"
#include "ZObject.hpp"

namespace z
{
    namespace math
    {
        /**
         * @brief 连杆位姿（位置和姿态）
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct LinkPose
        {
            /// @brief 位置 (世界坐标系)
            Vector<Scalar, 3> position = Vector<Scalar, 3>::zeros();
            /// @brief 姿态四元数 (XYZW 格式，世界坐标系)
            Vector<Scalar, 4> quaternion = { 0, 0, 0, 1 };

            /**
             * @brief 创建单位位姿
             */
            static LinkPose Identity()
            {
                return LinkPose();
            }
        };

        /**
         * @brief 关节状态
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct JointState
        {
            /// @brief 关节位置
            Scalar position = 0;
            /// @brief 关节速度
            Scalar velocity = 0;
            /// @brief 关节加速度
            Scalar acceleration = 0;
            /// @brief 关节力/力矩
            Scalar effort = 0;
        };

        /**
         * @brief 正向运动学模型
         * @tparam Scalar 标量类型 (float, double)
         * @tparam MaxJoints 最大关节数量（用于静态分配）
         */
        template<typename Scalar>
        class ForwardKinematics : public ZObject
        {
        public:
            using Vector3 = Vector<Scalar, 3>;
            using Vector4 = Vector<Scalar, 4>;
            using PoseType = LinkPose<Scalar>;
            using JointStateType = JointState<Scalar>;

            /**
             * @brief 默认构造函数
             */
            ForwardKinematics() = default;

            /**
             * @brief 从 URDF 解析器构建运动学模型
             * @param parser URDF 解析器
             */
            explicit ForwardKinematics(const URDFParser<Scalar>& parser)
            {
                BuildFromURDF(parser);
            }

            /**
             * @brief 从 URDF 解析器构建运动学模型
             * @param parser URDF 解析器
             * @return 是否成功
             */
            bool BuildFromURDF(const URDFParser<Scalar>& parser)
            {
                Clear();

                // 复制连杆信息
                for (const auto& [name, link] : parser.GetLinks())
                {
                    link_names_.push_back(name);
                    link_name_to_index_[name] = static_cast<int>(link_names_.size()) - 1;
                }

                // 获取根连杆
                root_link_ = parser.GetRootLinkName();
                if (root_link_.empty())
                {
                    std::cerr << "Failed to find root link" << std::endl;
                    return false;
                }

                // 构建关节列表和父子关系
                // 只考虑可驱动关节（revolute, continuous, prismatic）
                for (const auto& [name, joint] : parser.GetJoints())
                {
                    if (joint.type == JointType::Revolute ||
                        joint.type == JointType::Continuous ||
                        joint.type == JointType::Prismatic)
                    {
                        JointInfo info;
                        info.name = name;
                        info.type = joint.type;
                        info.parent_link = joint.parent_link;
                        info.child_link = joint.child_link;
                        info.axis = joint.axis;
                        info.origin_xyz = joint.origin_xyz;
                        info.origin_quaternion = z::math::quat_from_euler_xyz(joint.origin_rpy);

                        if (joint.limit.has_value())
                        {
                            info.has_limits = true;
                            info.lower_limit = joint.limit->lower;
                            info.upper_limit = joint.limit->upper;
                        }

                        joint_infos_.push_back(info);
                        joint_name_to_index_[name] = static_cast<int>(joint_infos_.size()) - 1;
                        actuated_joint_names_.push_back(name);
                    }
                    else if (joint.type == JointType::Fixed)
                    {
                        // 固定关节作为静态位姿存储
                        FixedJointInfo info;
                        info.parent_link = joint.parent_link;
                        info.child_link = joint.child_link;
                        info.position = joint.origin_xyz;
                        info.quaternion = quat_from_euler_xyz(joint.origin_rpy);
                        fixed_joints_.push_back(info);
                    }
                }

                // 构建连杆之间的变换关系
                BuildLinkTree();

                // 初始化关节状态
                joint_positions_.resize(actuated_joint_names_.size(), 0);
                link_poses_.resize(link_names_.size());

                return true;
            }

            /**
             * @brief 设置单个关节位置
             * @param joint_name 关节名称
             * @param position 关节位置 (rad 或 m)
             * @return 是否成功
             */
            bool SetJointPosition(const std::string& joint_name, Scalar position)
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it == joint_name_to_index_.end())
                {
                    return false;
                }

                int idx = it->second;

                // 检查限制
                const auto& info = joint_infos_[idx];
                if (info.has_limits && info.type != JointType::Continuous)
                {
                    position = std::clamp(position, info.lower_limit, info.upper_limit);
                }

                joint_positions_[idx] = position;
                return true;
            }

            /**
             * @brief 设置所有关节位置
             * @param positions 关节位置向量 (大小必须等于可驱动关节数量)
             * @return 是否成功
             */
            bool SetJointPositions(const std::vector<Scalar>& positions)
            {
                if (positions.size() != joint_positions_.size())
                {
                    std::cerr << "Joint position vector size mismatch: "
                        << positions.size() << " vs " << joint_positions_.size() << std::endl;
                    return false;
                }

                for (size_t i = 0; i < positions.size(); ++i)
                {
                    const auto& info = joint_infos_[i];
                    Scalar pos = positions[i];

                    // 检查限制
                    if (info.has_limits && info.type != JointType::Continuous)
                    {
                        pos = std::clamp(pos, info.lower_limit, info.upper_limit);
                    }

                    joint_positions_[i] = pos;
                }

                return true;
            }

            /**
             * @brief 设置关节位置（使用映射）
             * @param positions 关节名称到位置的映射
             */
            void SetJointPositions(const std::map<std::string, Scalar>& positions)
            {
                for (const auto& [name, pos] : positions)
                {
                    SetJointPosition(name, pos);
                }
            }

            /**
             * @brief 计算正向运动学
             * @return 是否成功
             */
            bool ComputeForwardKinematics()
            {
                if (link_names_.empty())
                {
                    return false;
                }

                // 初始化根连杆位姿（世界坐标系原点）
                int root_idx = GetLinkIndex(root_link_);
                if (root_idx < 0)
                {
                    return false;
                }

                link_poses_[root_idx] = PoseType::Identity();

                // 使用 BFS 遍历连杆树
                std::vector<bool> visited(link_names_.size(), false);
                std::queue<int> queue;
                queue.push(root_idx);
                visited[root_idx] = true;

                while (!queue.empty())
                {
                    int current_idx = queue.front();
                    queue.pop();

                    // 找到从当前连杆作为父连杆的所有关节
                    for (size_t j = 0; j < joint_infos_.size(); ++j)
                    {
                        const auto& joint = joint_infos_[j];
                        int parent_idx = GetLinkIndex(joint.parent_link);

                        if (parent_idx == current_idx)
                        {
                            int child_idx = GetLinkIndex(joint.child_link);
                            if (child_idx < 0 || visited[child_idx])
                            {
                                continue;
                            }

                            // 计算关节位姿（相对于父连杆）
                            PoseType joint_pose = ComputeJointPose(joint, joint_positions_[j]);

                            // 累积变换到世界坐标系
                            // R_world_child = R_world_parent * R_parent_child
                            // p_world_child = p_world_parent + R_world_parent * p_parent_child
                            const PoseType& parent_pose = link_poses_[parent_idx];
                            PoseType& child_pose = link_poses_[child_idx];

                            child_pose.quaternion = quat_mul(parent_pose.quaternion, joint_pose.quaternion);
                            child_pose.position = parent_pose.position +
                                quat_rotate(parent_pose.quaternion, joint_pose.position);

                            visited[child_idx] = true;
                            queue.push(child_idx);
                        }
                    }

                    // 处理固定关节
                    for (const auto& fixed : fixed_joints_)
                    {
                        int parent_idx = GetLinkIndex(fixed.parent_link);
                        if (parent_idx == current_idx)
                        {
                            int child_idx = GetLinkIndex(fixed.child_link);
                            if (child_idx < 0 || visited[child_idx])
                            {
                                continue;
                            }

                            const PoseType& parent_pose = link_poses_[parent_idx];
                            PoseType& child_pose = link_poses_[child_idx];

                            child_pose.quaternion = quat_mul(parent_pose.quaternion, fixed.quaternion);
                            child_pose.position = parent_pose.position +
                                quat_rotate(parent_pose.quaternion, fixed.position);

                            visited[child_idx] = true;
                            queue.push(child_idx);
                        }
                    }
                }

                return true;
            }

            /**
             * @brief 获取连杆位姿
             * @param link_name 连杆名称
             * @return 连杆位姿
             */
            PoseType GetLinkPose(const std::string& link_name) const
            {
                int idx = GetLinkIndex(link_name);
                if (idx < 0 || idx >= static_cast<int>(link_poses_.size()))
                {
                    throw std::runtime_error("Link not found: " + link_name);
                }
                return link_poses_[idx];
            }

            /**
             * @brief 安全地获取连杆位姿
             * @param link_name 连杆名称
             * @param pose 输出位姿
             * @return 是否成功
             */
            bool GetLinkPose(const std::string& link_name, PoseType& pose) const
            {
                int idx = GetLinkIndex(link_name);
                if (idx < 0 || idx >= static_cast<int>(link_poses_.size()))
                {
                    return false;
                }
                pose = link_poses_[idx];
                return true;
            }

            /**
             * @brief 获取所有连杆位姿
             * @return 连杆位姿映射表
             */
            std::map<std::string, PoseType> GetAllLinkPoses() const
            {
                std::map<std::string, PoseType> poses;
                for (size_t i = 0; i < link_names_.size(); ++i)
                {
                    poses[link_names_[i]] = link_poses_[i];
                }
                return poses;
            }

            /**
             * @brief 获取连杆在世界坐标系中的位置
             * @param link_name 连杆名称
             * @return 位置向量
             */
            Vector3 GetLinkPosition(const std::string& link_name) const
            {
                return GetLinkPose(link_name).position;
            }

            /**
             * @brief 获取连杆在世界坐标系中的姿态（四元数 XYZW）
             * @param link_name 连杆名称
             * @return 四元数
             */
            Vector4 GetLinkQuaternion(const std::string& link_name) const
            {
                return GetLinkPose(link_name).quaternion;
            }

            /**
             * @brief 获取当前关节位置
             * @return 关节位置向量
             */
            std::vector<Scalar> GetJointPositions() const
            {
                return joint_positions_;
            }

            /**
             * @brief 获取关节位置（通过名称）
             * @param joint_name 关节名称
             * @return 关节位置
             */
            Scalar GetJointPosition(const std::string& joint_name) const
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it == joint_name_to_index_.end())
                {
                    throw std::runtime_error("Joint not found: " + joint_name);
                }
                return joint_positions_[it->second];
            }

            /**
             * @brief 获取可驱动关节名称列表
             * @return 关节名称列表
             */
            const std::vector<std::string>& GetActuatedJointNames() const
            {
                return actuated_joint_names_;
            }

            /**
             * @brief 获取连杆名称列表
             * @return 连杆名称列表
             */
            const std::vector<std::string>& GetLinkNames() const
            {
                return link_names_;
            }

            /**
             * @brief 获取关节数量
             * @return 关节数量
             */
            size_t GetNumJoints() const
            {
                return joint_infos_.size();
            }

            /**
             * @brief 获取连杆数量
             * @return 连杆数量
             */
            size_t GetNumLinks() const
            {
                return link_names_.size();
            }

            /**
             * @brief 获取根连杆名称
             * @return 根连杆名称
             */
            const std::string& GetRootLinkName() const
            {
                return root_link_;
            }

            /**
             * @brief 检查关节是否存在
             * @param joint_name 关节名称
             * @return 是否存在
             */
            bool HasJoint(const std::string& joint_name) const
            {
                return joint_name_to_index_.find(joint_name) != joint_name_to_index_.end();
            }

            /**
             * @brief 获取关节轴（本地坐标系）
             * @param joint_name 关节名称
             * @return 关节轴
             */
            Vector3 GetJointAxis(const std::string& joint_name) const
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it != joint_name_to_index_.end())
                {
                    return joint_infos_[it->second].axis;
                }
                return Vector3{ 0, 0, 1 };
            }

            /**
             * @brief 获取关节类型
             * @param joint_name 关节名称
             * @return 关节类型
             */
            JointType GetJointType(const std::string& joint_name) const
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it != joint_name_to_index_.end())
                {
                    return joint_infos_[it->second].type;
                }
                return JointType::Unknown;
            }

            /**
             * @brief 获取关节原点位姿（相对于父连杆）
             * @param joint_name 关节名称
             * @return 位姿（位置和四元数）
             */
            PoseType GetJointOriginPose(const std::string& joint_name) const
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it != joint_name_to_index_.end())
                {
                    PoseType pose;
                    pose.position = joint_infos_[it->second].origin_xyz;
                    pose.quaternion = joint_infos_[it->second].origin_quaternion;
                    return pose;
                }
                return PoseType::Identity();
            }

            /**
             * @brief 获取关节父连杆名称
             * @param joint_name 关节名称
             * @return 父连杆名称
             */
            std::string GetJointParentLink(const std::string& joint_name) const
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it != joint_name_to_index_.end())
                {
                    return joint_infos_[it->second].parent_link;
                }
                return "";
            }

            /**
             * @brief 获取关节子连杆名称
             * @param joint_name 关节名称
             * @return 子连杆名称
             */
            std::string GetJointChildLink(const std::string& joint_name) const
            {
                auto it = joint_name_to_index_.find(joint_name);
                if (it != joint_name_to_index_.end())
                {
                    return joint_infos_[it->second].child_link;
                }
                return "";
            }

            /**
             * @brief 检查连杆是否存在
             * @param link_name 连杆名称
             * @return 是否存在
             */
            bool HasLink(const std::string& link_name) const
            {
                return link_name_to_index_.find(link_name) != link_name_to_index_.end();
            }

            /**
             * @brief 打印模型信息
             */
            void PrintModelInfo() const
            {
                std::cout << "=== Forward Kinematics Model ===" << std::endl;
                std::cout << "Root link: " << root_link_ << std::endl;
                std::cout << "Number of links: " << link_names_.size() << std::endl;
                std::cout << "Number of actuated joints: " << joint_infos_.size() << std::endl;
                std::cout << "Number of fixed joints: " << fixed_joints_.size() << std::endl;

                std::cout << "\nLinks:" << std::endl;
                for (const auto& name : link_names_)
                {
                    std::cout << "  - " << name << std::endl;
                }

                std::cout << "\nActuated Joints:" << std::endl;
                for (const auto& joint : joint_infos_)
                {
                    std::cout << "  - " << joint.name;
                    if (joint.has_limits && joint.type != JointType::Continuous)
                    {
                        std::cout << " [" << joint.lower_limit << ", " << joint.upper_limit << "]";
                    }
                    std::cout << std::endl;
                }

                std::cout << "\nFixed Joints:" << std::endl;
                for (const auto& joint : fixed_joints_)
                {
                    std::cout << "  - " << joint.parent_link << " -> " << joint.child_link << std::endl;
                }
                std::cout << "================================" << std::endl;
            }

            /**
             * @brief 清空模型
             */
            void Clear()
            {
                link_names_.clear();
                link_name_to_index_.clear();
                joint_infos_.clear();
                fixed_joints_.clear();
                joint_name_to_index_.clear();
                actuated_joint_names_.clear();
                joint_positions_.clear();
                link_poses_.clear();
                root_link_.clear();
            }

        private:
            /**
             * @brief 关节信息（内部使用）
             */
            struct JointInfo
            {
                std::string name;
                JointType type;
                std::string parent_link;
                std::string child_link;
                Vector3 axis;
                Vector3 origin_xyz;
                Vector4 origin_quaternion;
                bool has_limits = false;
                Scalar lower_limit = 0;
                Scalar upper_limit = 0;
            };

            /**
             * @brief 固定关节信息（内部使用）
             */
            struct FixedJointInfo
            {
                std::string parent_link;
                std::string child_link;
                Vector3 position;
                Vector4 quaternion;
            };

            std::vector<std::string> link_names_;
            std::unordered_map<std::string, int> link_name_to_index_;
            std::vector<JointInfo> joint_infos_;
            std::vector<FixedJointInfo> fixed_joints_;
            std::unordered_map<std::string, int> joint_name_to_index_;
            std::vector<std::string> actuated_joint_names_;
            std::vector<Scalar> joint_positions_;
            std::vector<PoseType> link_poses_;
            std::string root_link_;

            /**
             * @brief 获取连杆索引
             */
            int GetLinkIndex(const std::string& name) const
            {
                auto it = link_name_to_index_.find(name);
                if (it != link_name_to_index_.end())
                {
                    return it->second;
                }
                return -1;
            }

            /**
             * @brief 构建连杆树结构
             */
            void BuildLinkTree()
            {
                // 关节信息已经在构建时按父子关系存储
                // 这里可以添加额外的树结构验证
            }

            /**
             * @brief 计算关节位姿（相对于父连杆）
             * @param joint 关节信息
             * @param position 关节位置
             * @return 位姿（位置和四元数）
             */
            PoseType ComputeJointPose(const JointInfo& joint, Scalar position) const
            {
                PoseType pose;

                // 基础位姿（关节原点）
                pose.position = joint.origin_xyz;
                pose.quaternion = joint.origin_quaternion;

                // 根据关节类型计算运动部分的位姿
                switch (joint.type)
                {
                case JointType::Revolute:
                case JointType::Continuous:
                {
                    Vector3 axis_normalized = joint.axis;
                    Scalar axis_len = axis_normalized.length();
                    if (axis_len > 1e-10)
                    {
                        axis_normalized = axis_normalized / axis_len;
                    }
                    Vector3 so3 = axis_normalized * (position);
                    Vector4 motion_quat = so3_to_quat(so3);


                    pose.quaternion = quat_mul(joint.origin_quaternion, motion_quat);
                    break;
                }
                case JointType::Prismatic:
                {
                    // 平移关节：沿 axis 移动 position 距离
                    // axis 定义在 joint 坐标系中，需要归一化并转换到 parent 坐标系
                    Vector3 axis_normalized = joint.axis;
                    Scalar axis_len = axis_normalized.length();
                    if (axis_len > 1e-10)
                    {
                        axis_normalized = axis_normalized / axis_len;
                    }

                    Vector3 axis_in_parent = quat_rotate(joint.origin_quaternion, axis_normalized);
                    Vector3 translation = axis_in_parent * position;
                    pose.position = pose.position + translation;
                    break;
                }
                default:
                    break;
                }

                return pose;
            }

        };

        // 类型别名
        using ForwardKinematicsf = ForwardKinematics<float>;
        using ForwardKinematicsd = ForwardKinematics<double>;

    }  // namespace math
}  // namespace z
