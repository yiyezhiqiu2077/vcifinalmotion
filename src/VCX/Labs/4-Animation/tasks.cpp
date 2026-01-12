#include "Labs/4-Animation/tasks.h"
#include "CustomFunc.inl"
#include "IKSystem.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <spdlog/spdlog.h>

namespace VCX::Labs::Animation {

    void ForwardKinematics(IKSystem & ik, int StartIndex) {
        if (StartIndex == 0) {
            ik.JointGlobalRotation[0] = ik.JointLocalRotation[0];
            ik.JointGlobalPosition[0] = ik.JointLocalOffset[0];
            StartIndex                = 1;
        }

        for (int i = StartIndex; i < ik.JointLocalOffset.size(); i++) {
            // Task 1.1: Forward Kinematics
            // Global Rotation = Parent Global Rotation * Local Rotation
            ik.JointGlobalRotation[i] = ik.JointGlobalRotation[i - 1] * ik.JointLocalRotation[i];
            // Global Position = Parent Global Position + Parent Global Rotation * Local Offset
            ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i - 1] + ik.JointGlobalRotation[i - 1] * ik.JointLocalOffset[i];
        }
    }

    void InverseKinematicsCCD(IKSystem & ik, const glm::vec3 & EndPosition, int maxCCDIKIteration, float eps) {
        ForwardKinematics(ik, 0);
        int nJoints = ik.NumJoints();

        // Avoid infinite loops or unnecessary calculations if already close
        if (glm::l2Norm(ik.EndEffectorPosition() - EndPosition) <= eps) return;

        for (int iter = 0; iter < maxCCDIKIteration; iter++) {
            // Iterate from the second to last joint back to the root
            // (Assuming the last joint is the end-effector tip or fixed relative to it)
            for (int i = nJoints - 2; i >= 0; i--) {
                glm::vec3 currentEndPos = ik.EndEffectorPosition();
                float     error         = glm::l2Norm(currentEndPos - EndPosition);
                if (error < eps) return;

                glm::vec3 jointPos = ik.JointGlobalPosition[i];

                // Vector from joint to current end effector
                glm::vec3 toEnd = glm::normalize(currentEndPos - jointPos);
                // Vector from joint to target
                glm::vec3 toTarget = glm::normalize(EndPosition - jointPos);

                // Calculate rotation to align toEnd with toTarget
                // glm::rotation creates a quaternion that rotates vector A to vector B
                glm::quat deltaRot = glm::rotation(toEnd, toTarget);

                // Apply rotation. Note: We must modify LocalRotation.
                // The deltaRot is in World Space. We need to apply it carefully.
                // New Global = deltaRot * Old Global
                // Parent Global * New Local = deltaRot * Parent Global * Old Local
                // New Local = inverse(Parent Global) * deltaRot * Parent Global * Old Local

                glm::quat parentGlobalRot = (i == 0) ? glm::quat(1, 0, 0, 0) : ik.JointGlobalRotation[i - 1];
                glm::quat localUpdate     = glm::inverse(parentGlobalRot) * deltaRot * parentGlobalRot;

                ik.JointLocalRotation[i] = localUpdate * ik.JointLocalRotation[i];
                ik.JointLocalRotation[i] = glm::normalize(ik.JointLocalRotation[i]); // Numerical stability

                // Important: Update FK immediately so the next joint sees correct positions
                ForwardKinematics(ik, i);
            }
        }
    }

    void InverseKinematicsFABR(IKSystem & ik, const glm::vec3 & EndPosition, int maxFABRIKIteration, float eps) {
        ForwardKinematics(ik, 0);
        int                    nJoints = ik.NumJoints();
        std::vector<glm::vec3> backward_positions(nJoints, glm::vec3(0, 0, 0)), forward_positions(nJoints, glm::vec3(0, 0, 0));

        // Initialize forward_positions with current state
        for (int i = 0; i < nJoints; ++i) forward_positions[i] = ik.JointGlobalPosition[i];

        for (int IKIteration = 0; IKIteration < maxFABRIKIteration && glm::l2Norm(ik.EndEffectorPosition() - EndPosition) > eps; IKIteration++) {
            // Task 1.3: FABRIK IK

            // Backward Update: From End to Root
            backward_positions[nJoints - 1] = EndPosition; // Set end effector to target

            for (int i = nJoints - 2; i >= 0; i--) {
                // Find vector from new child position to current joint
                // We use backward_positions[i+1] (newly calculated) and forward_positions[i] (current estimate)
                glm::vec3 dir         = glm::normalize(forward_positions[i] - backward_positions[i + 1]);
                float     len         = ik.JointOffsetLength[i + 1]; // Length of bone between i and i+1
                backward_positions[i] = backward_positions[i + 1] + dir * len;
            }

            // Forward Update: From Root to End
            forward_positions[0] = ik.JointGlobalPosition[0]; // Root is fixed

            for (int i = 0; i < nJoints - 1; i++) {
                // Find vector from new parent position to current child
                glm::vec3 dir            = glm::normalize(backward_positions[i + 1] - forward_positions[i]);
                float     len            = ik.JointOffsetLength[i + 1];
                forward_positions[i + 1] = forward_positions[i] + dir * len;
            }

            ik.JointGlobalPosition = forward_positions;
        }

        // Re-compute rotations based on the new positions (Template provided logic)
        for (int i = 0; i < nJoints - 1; i++) {
            glm::vec3 oldDirLocal = glm::normalize(ik.JointLocalOffset[i + 1]);
            // We need to find the rotation that takes the default local offset direction
            // to the new direction in the parent's local frame.
            // But the provided template code uses global positions to deduce rotations.
            // Let's stick to the provided template logic flow for rotation reconstruction:
            ik.JointGlobalRotation[i] = glm::rotation(glm::normalize(ik.JointLocalOffset[i + 1]), glm::normalize(ik.JointGlobalPosition[i + 1] - ik.JointGlobalPosition[i]));
        }
        ik.JointLocalRotation[0] = ik.JointGlobalRotation[0];
        for (int i = 1; i < nJoints - 1; i++) {
            ik.JointLocalRotation[i] = glm::inverse(ik.JointGlobalRotation[i - 1]) * ik.JointGlobalRotation[i];
        }
        ForwardKinematics(ik, 0);
    }

    IKSystem::Vec3ArrPtr IKSystem::BuildCustomTargetPosition() {
        // Task 1.4: Bonus (Not Required/Optional)
        // Leaving default implementation
        int nums      = 5000;
        using Vec3Arr = std::vector<glm::vec3>;
        std::shared_ptr<Vec3Arr> custom(new Vec3Arr(nums));
        int                      index = 0;
        for (int i = 0; i < nums; i++) {
            float x_val = 1.5e-3f * custom_x(92 * glm::pi<float>() * i / nums);
            float y_val = 1.5e-3f * custom_y(92 * glm::pi<float>() * i / nums);
            if (std::abs(x_val) < 1e-3 || std::abs(y_val) < 1e-3) continue;
            (*custom)[index++] = glm::vec3(1.6f - x_val, 0.0f, y_val - 0.2f);
        }
        custom->resize(index);
        return custom;
    }

    static Eigen::VectorXf glm2eigen(std::vector<glm::vec3> const & glm_v) {
        Eigen::VectorXf v = Eigen::Map<Eigen::VectorXf const, Eigen::Aligned>(reinterpret_cast<float const *>(glm_v.data()), static_cast<int>(glm_v.size() * 3));
        return v;
    }

    static std::vector<glm::vec3> eigen2glm(Eigen::VectorXf const & eigen_v) {
        return std::vector<glm::vec3>(reinterpret_cast<glm::vec3 const *>(eigen_v.data()), reinterpret_cast<glm::vec3 const *>(eigen_v.data() + eigen_v.size()));
    }

    static Eigen::SparseMatrix<float> CreateEigenSparseMatrix(std::size_t n, std::vector<Eigen::Triplet<float>> const & triplets) {
        Eigen::SparseMatrix<float> matLinearized(n, n);
        matLinearized.setFromTriplets(triplets.begin(), triplets.end());
        return matLinearized;
    }

    static Eigen::VectorXf ComputeSimplicialLLT(Eigen::SparseMatrix<float> const & A, Eigen::VectorXf const & b) {
        auto solver = Eigen::SimplicialLLT<Eigen::SparseMatrix<float>>(A);
        return solver.solve(b);
    }

    void AdvanceMassSpringSystem(MassSpringSystem & system, float const dt) {
        // Task 2: Implicit Euler Mass-Spring System
        // Solve (M - h^2 * K) * v_new = M * v_old + h * f_total

        int const   steps = 1; // Implicit is stable, usually 1 step is fine, or fewer than explicit
        float const h     = dt / steps;

        int nParticles = system.Positions.size();
        int dim        = 3 * nParticles;

        for (std::size_t s = 0; s < steps; s++) {
            std::vector<Eigen::Triplet<float>> triplets;
            std::vector<glm::vec3>             forces(nParticles, glm::vec3(0, -system.Gravity * system.Mass, 0)); // Initialize with Gravity

            // 1. Accumulate Forces and Build Stiffness Matrix (K)
            // K is the Jacobian of Force w.r.t Position
            for (auto const spring : system.Springs) {
                auto const p0 = spring.AdjIdx.first;
                auto const p1 = spring.AdjIdx.second;

                glm::vec3 const x01 = system.Positions[p1] - system.Positions[p0];
                float const     len = glm::length(x01);

                if (len < 1e-6f) continue; // Avoid division by zero

                glm::vec3 const e01 = x01 / len; // Normalized direction

                // Explicit Force Calculation (for RHS)
                // Spring force + Damping force
                glm::vec3 const v01           = system.Velocities[p1] - system.Velocities[p0];
                float           f_spring_mag  = system.Stiffness * (len - spring.RestLength);
                float           f_damping_mag = system.Damping * glm::dot(v01, e01);
                glm::vec3       f             = (f_spring_mag + f_damping_mag) * e01;

                forces[p0] += f;
                forces[p1] -= f;

                // Jacobian (Stiffness Matrix K) Construction
                // We only use the stiffness part for the implicit matrix, usually ignoring damping Jacobian for simplicity
                // Hessian of Spring Potential: K_ij
                // K_block = k * [ (1 - L/|x|) * I + (L/|x|) * (x * x^T) / |x|^2 ]
                // Note: The formula for derivative of Force vector involves projectors.
                // dF_i / dx_j = k * ( (1 - L/l) * I + (L/l) * (x*xT) )

                glm::mat3 I   = glm::mat3(1.0f);
                glm::mat3 xxT = glm::outerProduct(e01, e01); // x * x^T / l^2 since e01 is normalized

                // Jacobian Block J = dF_p0 / dx_p1
                glm::mat3 J = system.Stiffness * ((1.0f - spring.RestLength / len) * I + (spring.RestLength / len) * xxT);

                // Add to triplets. The Matrix we solve is (M - h^2 * K).
                // dF0/dx0 = -J, dF0/dx1 = J
                // dF1/dx0 = J,  dF1/dx1 = -J
                // Global Matrix contributions (subtracting h^2 * K):
                // (p0, p0) -> - h^2 * (-J) = + h^2 * J
                // (p0, p1) -> - h^2 * (J)  = - h^2 * J
                // ...

                float h2 = h * h;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        float val = J[c][r] * h2; // Note: GLM is col-major, but indices are [col][row]. Triplet expects (row, col)

                        // Diagonal blocks (Add J)
                        triplets.emplace_back(3 * p0 + r, 3 * p0 + c, val);
                        triplets.emplace_back(3 * p1 + r, 3 * p1 + c, val);

                        // Off-diagonal blocks (Subtract J)
                        triplets.emplace_back(3 * p0 + r, 3 * p1 + c, -val);
                        triplets.emplace_back(3 * p1 + r, 3 * p0 + c, -val);
                    }
                }
            }

            // 2. Build Mass Matrix and RHS
            Eigen::VectorXf b(dim);

            for (int i = 0; i < nParticles; i++) {
                if (system.Fixed[i]) {
                    // Constraint: Velocity = 0
                    // Set Row to Identity, RHS to 0
                    triplets.emplace_back(3 * i + 0, 3 * i + 0, 1.0f);
                    triplets.emplace_back(3 * i + 1, 3 * i + 1, 1.0f);
                    triplets.emplace_back(3 * i + 2, 3 * i + 2, 1.0f);
                    b[3 * i + 0] = 0.0f;
                    b[3 * i + 1] = 0.0f;
                    b[3 * i + 2] = 0.0f;
                } else {
                    // Regular particle
                    // Matrix A diagonal += Mass
                    triplets.emplace_back(3 * i + 0, 3 * i + 0, system.Mass);
                    triplets.emplace_back(3 * i + 1, 3 * i + 1, system.Mass);
                    triplets.emplace_back(3 * i + 2, 3 * i + 2, system.Mass);

                    // RHS = M * v_old + h * f_total
                    glm::vec3 rhs_vec = system.Mass * system.Velocities[i] + h * forces[i];
                    b[3 * i + 0]      = rhs_vec.x;
                    b[3 * i + 1]      = rhs_vec.y;
                    b[3 * i + 2]      = rhs_vec.z;
                }
            }

            // 3. Solve Linear System
            auto            A     = CreateEigenSparseMatrix(dim, triplets);
            Eigen::VectorXf v_new = ComputeSimplicialLLT(A, b);

            // 4. Update State
            std::vector<glm::vec3> new_velocities = eigen2glm(v_new);
            system.Velocities                     = new_velocities;

            for (int i = 0; i < nParticles; ++i) {
                if (! system.Fixed[i]) {
                    system.Positions[i] += system.Velocities[i] * h;
                }
            }
        }
    }
} // namespace VCX::Labs::Animation