/**
 * @file so3.hpp
 * @author zishun zhou
 * @brief this file defines the SO3 class, which represents the special orthogonal group in 3D space.
 * @version 0.1
 * @date 2025-05-26
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "TensorType.hpp"
#include "VectorType.hpp"
#include <iostream>
#include <cmath>

namespace z
{
    namespace math
    {
        /**
         * @brief cross product of two 3D vectors.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param a 3D vector a
         * @param b 3D vector b
         * @return z::math::Vector<Scalar, 3> Resulting vector from the cross product of a and b.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 3> cross(const z::math::Vector<Scalar, 3>& a, const z::math::Vector<Scalar, 3>& b)
        {
            z::math::Vector<Scalar, 3> result;
            result[0] = a[1] * b[2] - a[2] * b[1];
            result[1] = a[2] * b[0] - a[0] * b[2];
            result[2] = a[0] * b[1] - a[1] * b[0];
            return result;
        }

        /**
         * @brief Compute the conjugate of a quaternion.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A quaternion represented as a vector in XYZW format.
         * @return z::math::Vector<Scalar, 4> Quaternion conjugate in XYZW format.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> quat_conjugate(const z::math::Vector<Scalar, 4>& quat)
        {
            return z::math::Vector<Scalar, 4>{ -quat[0], -quat[1], -quat[2], quat[3] };
        }

        /**
         * @brief Normalize a quaternion to unit length.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A quaternion represented as a vector in XYZW format.
         * @return z::math::Vector<Scalar, 4> Unit quaternion in XYZW format.
         * @throws std::runtime_error If the quaternion norm is zero, indicating it cannot be normalized.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> quat_unit(const z::math::Vector<Scalar, 4>& quat)
        {
            Scalar norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
            if (norm == 0) {
                throw std::runtime_error("Quaternion norm is zero, cannot normalize.");
            }
            return z::math::Vector<Scalar, 4>{ quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm };
        }

        /**
         * @brief Multiply two quaternions.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param a A quaternion represented as a vector in XYZW format.
         * @param b Another quaternion represented as a vector in XYZW format.
         * @return z::math::Vector<Scalar, 4> Quaternion resulting from the multiplication of a and b, in XYZW format.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> quat_mul(const z::math::Vector<Scalar, 4>& a, const z::math::Vector<Scalar, 4>& b)
        {
            Scalar x1 = a[0], y1 = a[1], z1 = a[2], w1 = a[3];
            Scalar x2 = b[0], y2 = b[1], z2 = b[2], w2 = b[3];

            Scalar ww = (z1 + x1) * (x2 + y2);
            Scalar yy = (w1 - y1) * (w2 + z2);
            Scalar zz = (w1 + y1) * (w2 - z2);
            Scalar xx = ww + yy + zz;
            Scalar qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));
            Scalar w = qq - ww + (z1 - y1) * (y2 - z2);
            Scalar x = qq - xx + (x1 + w1) * (x2 + w2);
            Scalar y = qq - yy + (w1 - x1) * (y2 + z2);
            Scalar z = qq - zz + (z1 + y1) * (w2 - x2);

            return z::math::Vector<Scalar, 4>{ x, y, z, w };
        }

        /**
         * @brief Rotate a 3D vector by a quaternion.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A quaternion represented as a vector in XYZW format.
         * @param vec A 3D vector to be rotated by the quaternion.
         * @return z::math::Vector<Scalar, 3> Resulting vector after rotation by the quaternion.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 3> quat_rotate(const z::math::Vector<Scalar, 4>& q, const z::math::Vector<Scalar, 3>& v)
        {
            Scalar q_w = q[3];
            z::math::Vector<Scalar, 3> q_vec = { q[0], q[1], q[2] };
            z::math::Vector<Scalar, 3> a = v * (2 * q_w * q_w - 1.0);
            z::math::Vector<Scalar, 3> b = cross(q_vec, v) * (2 * q_w);
            z::math::Vector<Scalar, 3> c = q_vec * (q_vec.dot(v)) * static_cast<Scalar>(2.0);
            return a + b + c;
        }

        /**
         * @brief Rotate a 3D vector by the inverse of a quaternion.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A quaternion represented as a vector in XYZW format.
         * @param vec A 3D vector to be rotated by the inverse of the quaternion.
         * @return z::math::Vector<Scalar, 3> Resulting vector after rotation by the inverse of the quaternion.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 3> quat_rotate_inverse(const z::math::Vector<Scalar, 4>& q, const z::math::Vector<Scalar, 3>& v)
        {
            Scalar q_w = q[3];
            z::math::Vector<Scalar, 3> q_vec = { q[0], q[1], q[2] };
            z::math::Vector<Scalar, 3> a = v * (2 * q_w * q_w - 1.0);
            z::math::Vector<Scalar, 3> b = cross(q_vec, v) * (2 * q_w);
            z::math::Vector<Scalar, 3> c = q_vec * (q_vec.dot(v)) * static_cast<Scalar>(2.0);
            return a - b + c;
        }

        /**
         * @brief Convert Euler angles (in radians) to quaternion representation.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param euler A vector containing the Euler angles in radians, in the order of (roll, pitch, yaw).
         * @return z::math::Vector<Scalar, 4> Quaternion representation(XYZW format) of the Euler angles.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> quat_from_euler_xyz(const z::math::Vector<Scalar, 3>& euler)
        {
            Scalar cy = std::cos(euler[2] * 0.5);
            Scalar sy = std::sin(euler[2] * 0.5);
            Scalar cp = std::cos(euler[1] * 0.5);
            Scalar sp = std::sin(euler[1] * 0.5);
            Scalar cr = std::cos(euler[0] * 0.5);
            Scalar sr = std::sin(euler[0] * 0.5);

            z::math::Vector<Scalar, 4> quat;
            Scalar qw, qx, qy, qz;
            qw = cy * cr * cp + sy * sr * sp;
            qx = cy * sr * cp - sy * cr * sp;
            qy = cy * cr * sp + sy * sr * cp;
            qz = sy * cr * cp - cy * sr * sp;
            quat = { qx, qy, qz, qw };

            return quat;
        }

        /**
         * @brief Convert quaternion representation to Euler angles (in radians).
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A vector containing the quaternion representation in XYZW format.
         * @return z::math::Vector<Scalar, 3> Euler angles in radians, in the order of (roll, pitch, yaw).
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 3> get_euler_xyz(const z::math::Vector<Scalar, 4>& quat)
        {
            Scalar qw = quat[3];
            Scalar qx = quat[0];
            Scalar qy = quat[1];
            Scalar qz = quat[2];

            Scalar roll = std::atan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz);
            Scalar pitch = std::asin(-2.0 * (qx * qz - qw * qy));
            Scalar yaw = std::atan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz);

            return z::math::Vector<Scalar, 3>{ roll, pitch, yaw };
        }


        /**
         * @brief Convert a quaternion to an so(3) vector representation.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A quaternion represented as a vector in XYZW format.
         * @return z::math::Vector<Scalar, 3> so(3) vector representation of the quaternion.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 3> so3_from_quat(const z::math::Vector<Scalar, 4>& quat)
        {
            auto unit_quat = quat_unit(quat);
            z::math::Vector<Scalar, 3> v = { unit_quat[0], unit_quat[1], unit_quat[2] };
            Scalar w = unit_quat[3];
            Scalar theta = 2 * std::acos(w);
            Scalar sin_theta = std::sin(theta / 2.0);
            if (std::abs(sin_theta) < 1e-6) {
                return z::math::Vector<Scalar, 3>{0, 0, 0}; // Return zero vector if sin(theta/2) is too small
            }
            return v * (theta / sin_theta);
        }

        /**
         * @brief Convert an so(3) vector representation to a quaternion.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param so3 A vector representing the so(3) representation of a rotation.
         * @return z::math::Vector<Scalar, 4> Quaternion representation of the so(3) vector in XYZW format.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> so3_to_quat(const z::math::Vector<Scalar, 3>& so3)
        {
            Scalar theta = so3.length();
            if (theta < 1e-6) {
                return z::math::Vector<Scalar, 4>{0, 0, 0, 1}; // Return identity quaternion if norm is too small
            }
            Scalar half_theta = theta / 2.0;
            Scalar sin_half_theta = std::sin(half_theta);
            return z::math::Vector<Scalar, 4>{ so3[0] * sin_half_theta / theta,
                so3[1] * sin_half_theta / theta,
                so3[2] * sin_half_theta / theta,
                std::cos(half_theta) };
        }


        /**
         * @brief Perform spherical linear interpolation (SLERP) between two quaternions.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param a quaternion a represented as begin interpolation quaternion in XYZW format.
         * @param b quaternion b represented as end interpolation quaternion in XYZW format.
         * @param t interpolation parameter, where t = 0 corresponds to a and t = 1 corresponds to b.
         * @return z::math::Vector<Scalar, 4>  Interpolated quaternion in XYZW format.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> quat_slerp(const z::math::Vector<Scalar, 4>& a, const z::math::Vector<Scalar, 4>& b, Scalar t)
        {
            // Ensure both quaternions are unit quaternions
            auto a_unit = quat_unit(a);
            auto b_unit = quat_unit(b);

            // Compute the dot product
            Scalar dot = a_unit.dot(b_unit);

            // If the dot product is negative, negate one quaternion to take the shortest path
            if (dot < 0.0) {
                b_unit = -b_unit;
                dot = -dot;
            }

            // Perform spherical linear interpolation
            Scalar theta = std::acos(dot);
            Scalar sin_theta = std::sin(theta);
            if (sin_theta < 1e-6) {
                // If sin(theta) is too small, return a linear interpolation
                return a_unit * (1 - t) + b_unit * t;
            }
            Scalar a_scale = std::sin((1 - t) * theta) / sin_theta;
            Scalar b_scale = std::sin(t * theta) / sin_theta;
            return quat_unit(a_unit * a_scale + b_unit * b_scale);
        }

        /**
         * @brief Compute the difference between two quaternions, ensuring the shortest path.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param a quaternion a represented as the first quaternion in XYZW format.
         * @param b quaternion b represented as the second quaternion in XYZW format.
         * @return z::math::Vector<Scalar, 4> Quaternion representing the difference between a and b, ensuring the shortest path.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 4> quat_diff(const z::math::Vector<Scalar, 4>& a, const z::math::Vector<Scalar, 4>& b)
        {
            auto quat_a = quat_unit(a);
            auto quat_b = quat_unit(b);
            Scalar dot = quat_a.dot(quat_b);
            if (dot < 0.0) {
                quat_b = -quat_b; // Ensure the shortest path
            }
            return quat_unit(quat_mul(quat_conjugate(a), b));
        }

        /**
         * @brief Compute the difference between two so(3) vectors, ensuring the shortest path.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param a so(3) vector a represented as the first vector in 3D space.
         * @param b so(3) vector b represented as the second vector in 3D space.
         * @return z::math::Vector<Scalar, 3> Vector representing the difference between a and b, ensuring the shortest path.
         */
        template<typename Scalar>
        z::math::Vector<Scalar, 3> so3_diff(const z::math::Vector<Scalar, 3>& a, const z::math::Vector<Scalar, 3>& b)
        {
            Scalar dot = a.dot(b);
            if (dot < 0.0) {
                return cross(b, a);
            }
            return cross(a, b);
        }

        /**
         * @brief Convert a quaternion to a rotation matrix.
         *
         * @tparam Scalar Type of the scalar (e.g., float, double).
         * @param quat A quaternion represented as a vector in XYZW format.
         * @return z::math::Tensor<Scalar, 3, 3> Rotation matrix corresponding to the quaternion.
         */
        template<typename Scalar>
        z::math::Tensor<Scalar, 3, 3> toRotationMatrix(const z::math::Vector<Scalar, 4>& quat)
        {
            Scalar qw = quat[3];
            Scalar qx = quat[0];
            Scalar qy = quat[1];
            Scalar qz = quat[2];

            z::math::Tensor<Scalar, 3, 3> R;


            R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
            R(0, 1) = 2 * (qx * qy - qz * qw);
            R(0, 2) = 2 * (qx * qz + qy * qw);

            R(1, 0) = 2 * (qx * qy + qz * qw);
            R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
            R(1, 2) = 2 * (qy * qz - qx * qw);

            R(2, 0) = 2 * (qx * qz - qy * qw);
            R(2, 1) = 2 * (qy * qz + qx * qw);
            R(2, 2) = 1 - 2 * (qx * qx + qy * qy);
            return R;
        }
    };
}