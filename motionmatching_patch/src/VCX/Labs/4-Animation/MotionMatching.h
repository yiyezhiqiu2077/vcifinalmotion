#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Labs/4-Animation/BVHLoader.h"

namespace VCX::Labs::Animation {

    struct GlobalPose {
        std::vector<glm::vec3> Pos;
        std::vector<glm::quat> Rot;
    };

    struct LocalPose {
        glm::vec3              RootTranslation { 0.f };
        std::vector<glm::quat> LocalRot; // LocalRot[0] is root local rotation
    };

    // Minimal, classic locomotion feature layout (26D):
    //  - Root velocity:            2  (x,z)
    //  - Future traj positions:    6  (3x (x,z))
    //  - Future facing directions: 6  (3x (x,z))
    //  - Left foot pos/vel:        6  (pos xyz + vel xyz)
    //  - Right foot pos/vel:       6  (pos xyz + vel xyz)
    static constexpr int kFeatureDim = 26;
    using Feature = std::array<float, kFeatureDim>;

    struct MatchResult {
        int   Index { 0 };
        float Cost  { 0.f };
    };

    struct MotionDatabase {
        BVHClip const * Clip { nullptr };
        int             LeftFoot  { -1 };
        int             RightFoot { -1 };

        float           FrameTime { 1.f / 60.f };
        int             FrameCount { 0 };

        std::vector<Feature> Features;   // per-frame
        std::vector<glm::vec3> RootPos;  // db-space global root position per frame
        std::vector<glm::quat> RootYaw;  // db-space yaw-only root rotation per frame

        void Build(BVHClip const & clip);

        MatchResult FindBestMatch(
            Feature const & query,
            Feature const & weights,
            int             currentIndex,
            int             excludeWindow) const;
    };

    // Sample BVH at a continuous time (seconds).
    LocalPose SampleLocalPose(BVHClip const & clip, float timeSeconds);

    // Forward kinematics to compute global positions/rotations in clip space.
    GlobalPose ComputeGlobalPose(BVHClip const & clip, LocalPose const & pose);

    // Build query feature from runtime information.
    //  - All vectors are in *world space*.
    //  - Internally it is converted to root-local feature space.
    Feature BuildQueryFeature(
        glm::vec3 const & rootPosWorld,
        glm::quat const & rootYawWorld,
        glm::vec3 const & rootVelWorld,
        std::array<glm::vec3, 3> const & desiredFuturePosWorld,
        std::array<glm::vec3, 3> const & desiredFutureFwdWorld,
        glm::vec3 const & leftFootPosWorld,
        glm::vec3 const & leftFootVelWorld,
        glm::vec3 const & rightFootPosWorld,
        glm::vec3 const & rightFootVelWorld);

} // namespace VCX::Labs::Animation
