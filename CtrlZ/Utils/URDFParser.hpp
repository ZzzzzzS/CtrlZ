/**
 * @file KinematicsParser.hpp
 * @brief URDF (Unified Robot Description Format) 解析器
 * @date 2026-03-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <optional>
#include <algorithm>
#include <cctype>
#include <set>

 // 使用项目自定义的数学类型
#include "MathTypes.hpp"

// 使用 tinyxml2 作为 XML 解析库
#include <tinyxml2.h>

namespace z
{
    namespace math
    {
        // 前向声明
        template<typename Scalar>
        class URDFParser;

        /**
         * @brief URDF 惯性属性
         * @tparam Scalar 标量类型 (float, double)
         */
        template<typename Scalar>
        struct Inertial
        {
            /// @brief 质量 (kg)
            Scalar mass = 0;
            /// @brief 质心位置 (相对于 link 坐标系)
            Vector<Scalar, 3> origin_xyz = Vector<Scalar, 3>::zeros();
            /// @brief 质心旋转 (RPY, 相对于 link 坐标系)
            Vector<Scalar, 3> origin_rpy = Vector<Scalar, 3>::zeros();
            /// @brief 惯性张量矩阵 (3x3)
            Tensor<Scalar, 3, 3> inertia;

            Inertial()
            {
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        inertia(i, j) = 0;
                    }
                }
            }
        };

        /**
         * @brief URDF 几何形状基类
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct Geometry
        {
            enum class Type
            {
                Box,
                Cylinder,
                Sphere,
                Mesh,
                Unknown
            };

            virtual ~Geometry() = default;
            virtual Type GetType() const = 0;
        };

        /**
         * @brief 盒子几何形状
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct BoxGeometry : public Geometry<Scalar>
        {
            Vector<Scalar, 3> size = Vector<Scalar, 3>::ones();
            typename Geometry<Scalar>::Type GetType() const override { return Geometry<Scalar>::Type::Box; }
        };

        /**
         * @brief 圆柱几何形状
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct CylinderGeometry : public Geometry<Scalar>
        {
            Scalar radius = 0;
            Scalar length = 0;
            typename Geometry<Scalar>::Type GetType() const override { return Geometry<Scalar>::Type::Cylinder; }
        };

        /**
         * @brief 球体几何形状
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct SphereGeometry : public Geometry<Scalar>
        {
            Scalar radius = 0;
            typename Geometry<Scalar>::Type GetType() const override { return Geometry<Scalar>::Type::Sphere; }
        };

        /**
         * @brief 网格几何形状
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct MeshGeometry : public Geometry<Scalar>
        {
            std::string filename;
            Vector<Scalar, 3> scale = Vector<Scalar, 3>::ones();
            typename Geometry<Scalar>::Type GetType() const override { return Geometry<Scalar>::Type::Mesh; }
        };

        /**
         * @brief 材质定义
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct Material
        {
            std::string name;
            Vector<Scalar, 4> color = { 0.8, 0.8, 0.8, 1.0 };  // RGBA
            std::string texture_filename;
        };

        /**
         * @brief 视觉属性
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct Visual
        {
            /// @brief 视觉原点位置
            Vector<Scalar, 3> origin_xyz = Vector<Scalar, 3>::zeros();
            /// @brief 视觉原点旋转 (RPY)
            Vector<Scalar, 3> origin_rpy = Vector<Scalar, 3>::zeros();
            /// @brief 几何形状
            std::shared_ptr<Geometry<Scalar>> geometry;
            /// @brief 材质
            std::optional<Material<Scalar>> material;
        };

        /**
         * @brief 碰撞属性
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct Collision
        {
            /// @brief 碰撞原点位置
            Vector<Scalar, 3> origin_xyz = Vector<Scalar, 3>::zeros();
            /// @brief 碰撞原点旋转 (RPY)
            Vector<Scalar, 3> origin_rpy = Vector<Scalar, 3>::zeros();
            /// @brief 几何形状
            std::shared_ptr<Geometry<Scalar>> geometry;
        };

        /**
         * @brief 连杆 (Link) 定义
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct Link
        {
            std::string name;
            std::optional<Inertial<Scalar>> inertial;
            std::vector<Visual<Scalar>> visuals;
            std::vector<Collision<Scalar>> collisions;
        };

        /**
         * @brief 关节类型
         */
        enum class JointType
        {
            Revolute,      // 旋转关节
            Continuous,    // 连续旋转关节
            Prismatic,     // 平移关节
            Fixed,         // 固定关节
            Floating,      // 浮动关节
            Planar,        // 平面关节
            Unknown
        };

        /**
         * @brief 关节限制
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct JointLimit
        {
            Scalar lower = 0;
            Scalar upper = 0;
            Scalar effort = 0;
            Scalar velocity = 0;
        };

        /**
         * @brief 动力学属性
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct JointDynamics
        {
            Scalar damping = 0;
            Scalar friction = 0;
        };

        /**
         * @brief 关节 (Joint) 定义
         * @tparam Scalar 标量类型
         */
        template<typename Scalar>
        struct Joint
        {
            std::string name;
            JointType type = JointType::Unknown;
            std::string parent_link;
            std::string child_link;
            /// @brief 关节原点位置 (相对于 parent link)
            Vector<Scalar, 3> origin_xyz = Vector<Scalar, 3>::zeros();
            /// @brief 关节原点旋转 (RPY, 相对于 parent link)
            Vector<Scalar, 3> origin_rpy = Vector<Scalar, 3>::zeros();
            /// @brief 旋转/平移轴
            Vector<Scalar, 3> axis = { 0, 0, 1 };
            std::optional<JointLimit<Scalar>> limit;
            std::optional<JointDynamics<Scalar>> dynamics;
        };

        /**
         * @brief URDF 解析器
         * @tparam Scalar 标量类型 (float, double)
         */
        template<typename Scalar>
        class URDFParser
        {
        public:
            using LinkType = Link<Scalar>;
            using JointType_ = Joint<Scalar>;
            using Vector3 = Vector<Scalar, 3>;

            /**
             * @brief 从文件加载 URDF
             * @param filename URDF 文件路径
             * @return 是否成功
             */
            bool LoadFromFile(const std::string& filename)
            {
                tinyxml2::XMLDocument doc;
                if (doc.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS)
                {
                    std::cerr << "Failed to load URDF file: " << filename << std::endl;
                    return false;
                }
                return ParseDocument(doc);
            }

            /**
             * @brief 从字符串加载 URDF
             * @param xml_string URDF XML 字符串
             * @return 是否成功
             */
            bool LoadFromString(const std::string& xml_string)
            {
                tinyxml2::XMLDocument doc;
                if (doc.Parse(xml_string.c_str()) != tinyxml2::XML_SUCCESS)
                {
                    std::cerr << "Failed to parse URDF XML string" << std::endl;
                    return false;
                }
                return ParseDocument(doc);
            }

            /**
             * @brief 获取所有连杆
             * @return 连杆列表 (name, Link) 按 DFS 顺序
             */
            const std::vector<std::pair<std::string, LinkType>>& GetLinks() const
            {
                return links_;
            }

            /**
             * @brief 获取所有关节
             * @return 关节列表 (name, Joint) 按 DFS 顺序
             */
            const std::vector<std::pair<std::string, JointType_>>& GetJoints() const
            {
                return joints_;
            }

            /**
             * @brief 获取指定名称的连杆
             * @param name 连杆名称
             * @return 连杆指针，如果不存在返回 nullptr
             */
            const LinkType* GetLink(const std::string& name) const
            {
                auto it = link_name_to_index_.find(name);
                if (it != link_name_to_index_.end() && it->second < links_.size())
                    return &(links_[it->second].second);
                return nullptr;
            }

            /**
             * @brief 获取指定名称的关节
             * @param name 关节名称
             * @return 关节指针，如果不存在返回 nullptr
             */
            const JointType_* GetJoint(const std::string& name) const
            {
                auto it = joint_name_to_index_.find(name);
                if (it != joint_name_to_index_.end() && it->second < joints_.size())
                    return &(joints_[it->second].second);
                return nullptr;
            }

            /**
             * @brief 获取机器人名称
             * @return 机器人名称
             */
            const std::string& GetRobotName() const
            {
                return robot_name_;
            }

            /**
             * @brief 获取根连杆名称
             * @return 根连杆名称
             */
            std::string GetRootLinkName() const
            {
                // 找到没有作为 child link 的 link
                std::set<std::string> child_links;
                for (const auto& [name, joint] : joints_)
                {
                    child_links.insert(joint.child_link);
                }

                for (const auto& [name, link] : links_)
                {
                    if (child_links.find(name) == child_links.end())
                        return name;
                }
                return "";
            }

            /**
             * @brief 获取指定 link 在 DFS 序列中的索引
             * @param name 连杆名称
             * @return 索引，如果不存在返回 size_t的最大值
             */
            size_t GetLinkIndex(const std::string& name) const
            {
                auto it = link_name_to_index_.find(name);
                if (it != link_name_to_index_.end())
                    return it->second;
                return static_cast<size_t>(-1);
            }

            /**
             * @brief 获取指定 joint 在 DFS 序列中的索引
             * @param name 关节名称
             * @return 索引，如果不存在返回 size_t的最大值
             */
            size_t GetJointIndex(const std::string& name) const
            {
                auto it = joint_name_to_index_.find(name);
                if (it != joint_name_to_index_.end())
                    return it->second;
                return static_cast<size_t>(-1);
            }


            /**
             * @brief 获取所有可驱动关节名称
             * @return 关节名称列表 (按 DFS 顺序)
             */
            std::vector<std::string> GetActuatedJointNames() const
            {
                std::vector<std::string> names;
                for (const auto& [name, joint] : joints_)
                {
                    if (joint.type == JointType::Revolute ||
                        joint.type == JointType::Prismatic ||
                        joint.type == JointType::Continuous)
                    {
                        names.push_back(name);
                    }
                }
                return names;
            }

            /**
             * @brief 打印机器人信息（用于调试）
             */
            void PrintRobotInfo() const
            {
                std::cout << "Robot Name: " << robot_name_ << std::endl;
                std::cout << "Root Link: " << GetRootLinkName() << std::endl;
                std::cout << "\nLinks (" << links_.size() << ") [DFS Order]:" << std::endl;
                for (const auto& [name, link] : links_)
                {
                    std::cout << "  - " << name;
                    if (link.inertial.has_value())
                        std::cout << " [inertial]";
                    std::cout << " (visuals: " << link.visuals.size()
                        << ", collisions: " << link.collisions.size() << ")";
                    std::cout << std::endl;
                }

                std::cout << "\nJoints (" << joints_.size() << ") [DFS Order]:" << std::endl;
                for (const auto& [name, joint] : joints_)
                {
                    std::cout << "  - " << name << " ("
                        << JointTypeToString(joint.type) << "): "
                        << joint.parent_link << " -> " << joint.child_link << std::endl;
                }
            }

        private:
            std::string robot_name_;
            // 使用 vector 保存 DFS 顺序，同时维护 name 到 index 的映射
            std::vector<std::pair<std::string, LinkType>> links_;
            std::vector<std::pair<std::string, JointType_>> joints_;
            std::unordered_map<std::string, size_t> link_name_to_index_;
            std::unordered_map<std::string, size_t> joint_name_to_index_;

            // 用于解析阶段的临时存储
            std::map<std::string, LinkType> temp_links_;
            std::map<std::string, JointType_> temp_joints_;

            bool ParseDocument(tinyxml2::XMLDocument& doc)
            {
                tinyxml2::XMLElement* robot_elem = doc.FirstChildElement("robot");
                if (!robot_elem)
                {
                    std::cerr << "No <robot> element found in URDF" << std::endl;
                    return false;
                }

                const char* name_attr = robot_elem->Attribute("name");
                if (name_attr)
                    robot_name_ = name_attr;

                // 清空临时存储
                temp_links_.clear();
                temp_joints_.clear();
                links_.clear();
                joints_.clear();
                link_name_to_index_.clear();
                joint_name_to_index_.clear();

                // 解析材质
                std::map<std::string, Material<Scalar>> materials;
                for (tinyxml2::XMLElement* mat_elem = robot_elem->FirstChildElement("material");
                    mat_elem;
                    mat_elem = mat_elem->NextSiblingElement("material"))
                {
                    ParseMaterial(mat_elem, materials);
                }

                // 解析连杆
                for (tinyxml2::XMLElement* link_elem = robot_elem->FirstChildElement("link");
                    link_elem;
                    link_elem = link_elem->NextSiblingElement("link"))
                {
                    ParseLink(link_elem, materials);
                }

                // 解析关节
                for (tinyxml2::XMLElement* joint_elem = robot_elem->FirstChildElement("joint");
                    joint_elem;
                    joint_elem = joint_elem->NextSiblingElement("joint"))
                {
                    ParseJoint(joint_elem);
                }

                // 按照 DFS 顺序排序 links 和 joints
                ReorderByDFS();

                // 清空临时存储
                temp_links_.clear();
                temp_joints_.clear();

                return true;
            }

            /**
             * @brief 按照 DFS 顺序重新排序 links 和 joints
             */
            void ReorderByDFS()
            {
                // 建立 parent_link -> [(joint_name, child_link)] 的映射
                std::map<std::string, std::vector<std::pair<std::string, std::string>>> parent_to_children;
                for (const auto& [name, joint] : temp_joints_)
                {
                    parent_to_children[joint.parent_link].push_back({ name, joint.child_link });
                }

                // 找到 root link
                std::string root_link = GetRootLinkNameFromTemp();
                if (root_link.empty() && !temp_links_.empty())
                {
                    root_link = temp_links_.begin()->first;
                }

                // DFS 遍历
                std::set<std::string> visited_links;
                std::set<std::string> added_joints;

                std::function<void(const std::string&)> dfs = [&](const std::string& link_name)
                    {
                        if (visited_links.count(link_name))
                            return;

                        visited_links.insert(link_name);

                        // 添加当前 link
                        auto link_it = temp_links_.find(link_name);
                        if (link_it != temp_links_.end())
                        {
                            link_name_to_index_[link_name] = links_.size();
                            links_.push_back({ link_name, link_it->second });
                        }

                        // 遍历以当前 link 为 parent 的 joints
                        auto it = parent_to_children.find(link_name);
                        if (it != parent_to_children.end())
                        {
                            for (const auto& [joint_name, child_link] : it->second)
                            {
                                if (!added_joints.count(joint_name))
                                {
                                    added_joints.insert(joint_name);
                                    auto joint_it = temp_joints_.find(joint_name);
                                    if (joint_it != temp_joints_.end())
                                    {
                                        joint_name_to_index_[joint_name] = joints_.size();
                                        joints_.push_back({ joint_name, joint_it->second });
                                    }
                                    dfs(child_link);
                                }
                            }
                        }
                    };

                if (!root_link.empty())
                {
                    dfs(root_link);
                }

                // 添加任何未被访问的 links 和 joints（防止遗漏孤立节点）
                for (const auto& [name, link] : temp_links_)
                {
                    if (!visited_links.count(name))
                    {
                        link_name_to_index_[name] = links_.size();
                        links_.push_back({ name, link });
                    }
                }
                for (const auto& [name, joint] : temp_joints_)
                {
                    if (!added_joints.count(name))
                    {
                        joint_name_to_index_[name] = joints_.size();
                        joints_.push_back({ name, joint });
                    }
                }
            }

            /**
             * @brief 从临时数据中获取 root link 名称
             */
            std::string GetRootLinkNameFromTemp() const
            {
                std::set<std::string> child_links;
                for (const auto& [name, joint] : temp_joints_)
                {
                    child_links.insert(joint.child_link);
                }

                for (const auto& [name, link] : temp_links_)
                {
                    if (child_links.find(name) == child_links.end())
                        return name;
                }
                return "";
            }

            void ParseMaterial(tinyxml2::XMLElement* elem,
                std::map<std::string, Material<Scalar>>& materials)
            {
                const char* name_attr = elem->Attribute("name");
                if (!name_attr)
                    return;

                Material<Scalar> mat;
                mat.name = name_attr;

                tinyxml2::XMLElement* color_elem = elem->FirstChildElement("color");
                if (color_elem)
                {
                    const char* rgba_attr = color_elem->Attribute("rgba");
                    if (rgba_attr)
                    {
                        mat.color = ParseVector4(rgba_attr);
                    }
                }

                tinyxml2::XMLElement* texture_elem = elem->FirstChildElement("texture");
                if (texture_elem)
                {
                    const char* filename_attr = texture_elem->Attribute("filename");
                    if (filename_attr)
                        mat.texture_filename = filename_attr;
                }

                materials[mat.name] = mat;
            }

            void ParseLink(tinyxml2::XMLElement* elem,
                const std::map<std::string, Material<Scalar>>& materials)
            {
                const char* name_attr = elem->Attribute("name");
                if (!name_attr)
                    return;

                LinkType link;
                link.name = name_attr;

                // 解析 inertial
                tinyxml2::XMLElement* inertial_elem = elem->FirstChildElement("inertial");
                if (inertial_elem)
                {
                    link.inertial = ParseInertial(inertial_elem);
                }

                // 解析 visual
                for (tinyxml2::XMLElement* visual_elem = elem->FirstChildElement("visual");
                    visual_elem;
                    visual_elem = visual_elem->NextSiblingElement("visual"))
                {
                    link.visuals.push_back(ParseVisual(visual_elem, materials));
                }

                // 解析 collision
                for (tinyxml2::XMLElement* collision_elem = elem->FirstChildElement("collision");
                    collision_elem;
                    collision_elem = collision_elem->NextSiblingElement("collision"))
                {
                    link.collisions.push_back(ParseCollision(collision_elem));
                }

                temp_links_[link.name] = std::move(link);
            }

            Inertial<Scalar> ParseInertial(tinyxml2::XMLElement* elem)
            {
                Inertial<Scalar> inertial;

                // 解析 origin
                tinyxml2::XMLElement* origin_elem = elem->FirstChildElement("origin");
                if (origin_elem)
                {
                    const char* xyz_attr = origin_elem->Attribute("xyz");
                    if (xyz_attr)
                        inertial.origin_xyz = ParseVector3(xyz_attr);

                    const char* rpy_attr = origin_elem->Attribute("rpy");
                    if (rpy_attr)
                        inertial.origin_rpy = ParseVector3(rpy_attr);
                }

                // 解析 mass
                tinyxml2::XMLElement* mass_elem = elem->FirstChildElement("mass");
                if (mass_elem)
                {
                    const char* value_attr = mass_elem->Attribute("value");
                    if (value_attr)
                        inertial.mass = static_cast<Scalar>(std::stod(value_attr));
                }

                // 解析 inertia
                tinyxml2::XMLElement* inertia_elem = elem->FirstChildElement("inertia");
                if (inertia_elem)
                {
                    inertial.inertia = ParseInertiaMatrix(inertia_elem);
                }

                return inertial;
            }

            Tensor<Scalar, 3, 3> ParseInertiaMatrix(tinyxml2::XMLElement* elem)
            {
                Tensor<Scalar, 3, 3> inertia;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        inertia(i, j) = 0;

                auto get_attr = [&](const char* name) -> Scalar {
                    const char* val = elem->Attribute(name);
                    if (val)
                        return static_cast<Scalar>(std::stod(val));
                    return 0;
                    };

                inertia(0, 0) = get_attr("ixx");
                inertia(0, 1) = get_attr("ixy");
                inertia(0, 2) = get_attr("ixz");
                inertia(1, 0) = inertia(0, 1);
                inertia(1, 1) = get_attr("iyy");
                inertia(1, 2) = get_attr("iyz");
                inertia(2, 0) = inertia(0, 2);
                inertia(2, 1) = inertia(1, 2);
                inertia(2, 2) = get_attr("izz");

                return inertia;
            }

            Visual<Scalar> ParseVisual(tinyxml2::XMLElement* elem,
                const std::map<std::string, Material<Scalar>>& materials)
            {
                Visual<Scalar> visual;

                // 解析 origin
                tinyxml2::XMLElement* origin_elem = elem->FirstChildElement("origin");
                if (origin_elem)
                {
                    const char* xyz_attr = origin_elem->Attribute("xyz");
                    if (xyz_attr)
                        visual.origin_xyz = ParseVector3(xyz_attr);

                    const char* rpy_attr = origin_elem->Attribute("rpy");
                    if (rpy_attr)
                        visual.origin_rpy = ParseVector3(rpy_attr);
                }

                // 解析 geometry
                tinyxml2::XMLElement* geometry_elem = elem->FirstChildElement("geometry");
                if (geometry_elem)
                {
                    visual.geometry = ParseGeometry(geometry_elem);
                }

                // 解析 material
                tinyxml2::XMLElement* material_elem = elem->FirstChildElement("material");
                if (material_elem)
                {
                    const char* name_attr = material_elem->Attribute("name");
                    if (name_attr)
                    {
                        auto it = materials.find(name_attr);
                        if (it != materials.end())
                        {
                            visual.material = it->second;
                        }
                        else
                        {
                            // 内联 material 定义
                            Material<Scalar> mat;
                            mat.name = name_attr;
                            tinyxml2::XMLElement* color_elem = material_elem->FirstChildElement("color");
                            if (color_elem)
                            {
                                const char* rgba_attr = color_elem->Attribute("rgba");
                                if (rgba_attr)
                                    mat.color = ParseVector4(rgba_attr);
                            }
                            visual.material = mat;
                        }
                    }
                }

                return visual;
            }

            Collision<Scalar> ParseCollision(tinyxml2::XMLElement* elem)
            {
                Collision<Scalar> collision;

                // 解析 origin
                tinyxml2::XMLElement* origin_elem = elem->FirstChildElement("origin");
                if (origin_elem)
                {
                    const char* xyz_attr = origin_elem->Attribute("xyz");
                    if (xyz_attr)
                        collision.origin_xyz = ParseVector3(xyz_attr);

                    const char* rpy_attr = origin_elem->Attribute("rpy");
                    if (rpy_attr)
                        collision.origin_rpy = ParseVector3(rpy_attr);
                }

                // 解析 geometry
                tinyxml2::XMLElement* geometry_elem = elem->FirstChildElement("geometry");
                if (geometry_elem)
                {
                    collision.geometry = ParseGeometry(geometry_elem);
                }

                return collision;
            }

            std::shared_ptr<Geometry<Scalar>> ParseGeometry(tinyxml2::XMLElement* elem)
            {
                // Box
                if (tinyxml2::XMLElement* box_elem = elem->FirstChildElement("box"))
                {
                    auto box = std::make_shared<BoxGeometry<Scalar>>();
                    const char* size_attr = box_elem->Attribute("size");
                    if (size_attr)
                        box->size = ParseVector3(size_attr);
                    return box;
                }

                // Cylinder
                if (tinyxml2::XMLElement* cyl_elem = elem->FirstChildElement("cylinder"))
                {
                    auto cyl = std::make_shared<CylinderGeometry<Scalar>>();
                    const char* radius_attr = cyl_elem->Attribute("radius");
                    if (radius_attr)
                        cyl->radius = static_cast<Scalar>(std::stod(radius_attr));
                    const char* length_attr = cyl_elem->Attribute("length");
                    if (length_attr)
                        cyl->length = static_cast<Scalar>(std::stod(length_attr));
                    return cyl;
                }

                // Sphere
                if (tinyxml2::XMLElement* sph_elem = elem->FirstChildElement("sphere"))
                {
                    auto sph = std::make_shared<SphereGeometry<Scalar>>();
                    const char* radius_attr = sph_elem->Attribute("radius");
                    if (radius_attr)
                        sph->radius = static_cast<Scalar>(std::stod(radius_attr));
                    return sph;
                }

                // Mesh
                if (tinyxml2::XMLElement* mesh_elem = elem->FirstChildElement("mesh"))
                {
                    auto mesh = std::make_shared<MeshGeometry<Scalar>>();
                    const char* filename_attr = mesh_elem->Attribute("filename");
                    if (filename_attr)
                        mesh->filename = filename_attr;
                    const char* scale_attr = mesh_elem->Attribute("scale");
                    if (scale_attr)
                        mesh->scale = ParseVector3(scale_attr);
                    return mesh;
                }

                return nullptr;
            }

            void ParseJoint(tinyxml2::XMLElement* elem)
            {
                const char* name_attr = elem->Attribute("name");
                if (!name_attr)
                    return;

                JointType_ joint;
                joint.name = name_attr;

                // 解析 type
                const char* type_attr = elem->Attribute("type");
                if (type_attr)
                    joint.type = StringToJointType(type_attr);

                // 解析 parent
                tinyxml2::XMLElement* parent_elem = elem->FirstChildElement("parent");
                if (parent_elem)
                {
                    const char* link_attr = parent_elem->Attribute("link");
                    if (link_attr)
                        joint.parent_link = link_attr;
                }

                // 解析 child
                tinyxml2::XMLElement* child_elem = elem->FirstChildElement("child");
                if (child_elem)
                {
                    const char* link_attr = child_elem->Attribute("link");
                    if (link_attr)
                        joint.child_link = link_attr;
                }

                // 解析 origin
                tinyxml2::XMLElement* origin_elem = elem->FirstChildElement("origin");
                if (origin_elem)
                {
                    const char* xyz_attr = origin_elem->Attribute("xyz");
                    if (xyz_attr)
                        joint.origin_xyz = ParseVector3(xyz_attr);

                    const char* rpy_attr = origin_elem->Attribute("rpy");
                    if (rpy_attr)
                        joint.origin_rpy = ParseVector3(rpy_attr);
                }

                // 解析 axis
                tinyxml2::XMLElement* axis_elem = elem->FirstChildElement("axis");
                if (axis_elem)
                {
                    const char* xyz_attr = axis_elem->Attribute("xyz");
                    if (xyz_attr)
                        joint.axis = ParseVector3(xyz_attr);
                }

                // 解析 limit
                tinyxml2::XMLElement* limit_elem = elem->FirstChildElement("limit");
                if (limit_elem)
                {
                    JointLimit<Scalar> limit;
                    const char* lower_attr = limit_elem->Attribute("lower");
                    if (lower_attr)
                        limit.lower = static_cast<Scalar>(std::stod(lower_attr));
                    const char* upper_attr = limit_elem->Attribute("upper");
                    if (upper_attr)
                        limit.upper = static_cast<Scalar>(std::stod(upper_attr));
                    const char* effort_attr = limit_elem->Attribute("effort");
                    if (effort_attr)
                        limit.effort = static_cast<Scalar>(std::stod(effort_attr));
                    const char* velocity_attr = limit_elem->Attribute("velocity");
                    if (velocity_attr)
                        limit.velocity = static_cast<Scalar>(std::stod(velocity_attr));
                    joint.limit = limit;
                }

                // 解析 dynamics
                tinyxml2::XMLElement* dynamics_elem = elem->FirstChildElement("dynamics");
                if (dynamics_elem)
                {
                    JointDynamics<Scalar> dynamics;
                    const char* damping_attr = dynamics_elem->Attribute("damping");
                    if (damping_attr)
                        dynamics.damping = static_cast<Scalar>(std::stod(damping_attr));
                    const char* friction_attr = dynamics_elem->Attribute("friction");
                    if (friction_attr)
                        dynamics.friction = static_cast<Scalar>(std::stod(friction_attr));
                    joint.dynamics = dynamics;
                }

                temp_joints_[joint.name] = std::move(joint);
            }

            Vector<Scalar, 3> ParseVector3(const std::string& str)
            {
                Vector<Scalar, 3> vec = Vector<Scalar, 3>::zeros();
                std::istringstream iss(str);
                for (int i = 0; i < 3; ++i)
                {
                    if (!(iss >> vec[i]))
                        break;
                }
                return vec;
            }

            Vector<Scalar, 4> ParseVector4(const std::string& str)
            {
                Vector<Scalar, 4> vec = Vector<Scalar, 4>::zeros();
                std::istringstream iss(str);
                for (int i = 0; i < 4; ++i)
                {
                    if (!(iss >> vec[i]))
                        break;
                }
                return vec;
            }

            JointType StringToJointType(const std::string& str)
            {
                std::string lower = str;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

                if (lower == "revolute")
                    return JointType::Revolute;
                if (lower == "continuous")
                    return JointType::Continuous;
                if (lower == "prismatic")
                    return JointType::Prismatic;
                if (lower == "fixed")
                    return JointType::Fixed;
                if (lower == "floating")
                    return JointType::Floating;
                if (lower == "planar")
                    return JointType::Planar;

                return JointType::Unknown;
            }

            std::string JointTypeToString(JointType type) const
            {
                switch (type)
                {
                case JointType::Revolute:   return "revolute";
                case JointType::Continuous: return "continuous";
                case JointType::Prismatic:  return "prismatic";
                case JointType::Fixed:      return "fixed";
                case JointType::Floating:   return "floating";
                case JointType::Planar:     return "planar";
                default:                    return "unknown";
                }
            }
        };

        // 类型别名
        using URDFParserf = URDFParser<float>;
        using URDFParserd = URDFParser<double>;

    }  // namespace math
}  // namespace z
