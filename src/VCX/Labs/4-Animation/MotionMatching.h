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

    // Feature layout (extended):
    //  - Root velocity:            2  (x,z)
    //  - Future traj positions:    6  (3x (x,z))
    //  - Future facing directions: 6  (3x (x,z))
    //  - Left foot pos/vel:        6  (pos xyz + vel xyz)
    //  - Right foot pos/vel:       6  (pos xyz + vel xyz)
    //  - Head pos/vel:             6
    //  - Left hand pos/vel:        6
    //  - Right hand pos/vel:       6
    //  - Foot contact (L,R):       2  (0/1)
    static constexpr int kFeatureDim = 46;
    using Feature                    = std::array<float, kFeatureDim>;

    struct MatchResult {
        int   Index { 0 };
        float Cost { 0.f };
    };

    struct MotionDatabase {
        BVHClip const * Clip { nullptr };

        int LeftFoot { -1 };
        int RightFoot { -1 };
        int Head { -1 };
        int LeftHand { -1 };
        int RightHand { -1 };

        float FrameTime { 1.f / 60.f };
        int   FrameCount { 0 };

        std::vector<Feature>   Features; // per-frame
        std::vector<glm::vec3> RootPos;  // db-space global root position per frame
        std::vector<glm::quat> RootYaw;  // db-space yaw-only root rotation per frame

        // Per-dim scale to reduce unit/variance issues.
        Feature InvStd {}; // 1/sigma, clamped

        // Simple accel: stride for coarse search (1 = full scan)
        int SearchStride { 4 };

        void Build(BVHClip const & clip);

        MatchResult FindBestMatch(
            Feature const & query,
            Feature const & weights,
            int             currentIndex,
            int             excludeWindow) const;
    };

    LocalPose  SampleLocalPose(BVHClip const & clip, float timeSeconds);
    GlobalPose ComputeGlobalPose(BVHClip const & clip, LocalPose const & pose);

    // Build query feature from runtime information (world space in, internally root-local).
    Feature BuildQueryFeature(
        glm::vec3 const &                rootPosWorld,
        glm::quat const &                rootYawWorld,
        glm::vec3 const &                rootVelWorld,
        std::array<glm::vec3, 3> const & desiredFuturePosWorld,
        std::array<glm::vec3, 3> const & desiredFutureFwdWorld,
        glm::vec3 const &                leftFootPosWorld,
        glm::vec3 const &                leftFootVelWorld,
        glm::vec3 const &                rightFootPosWorld,
        glm::vec3 const &                rightFootVelWorld,
        glm::vec3 const &                headPosWorld      = glm::vec3(0.f),
        glm::vec3 const &                headVelWorld      = glm::vec3(0.f),
        glm::vec3 const &                leftHandPosWorld  = glm::vec3(0.f),
        glm::vec3 const &                leftHandVelWorld  = glm::vec3(0.f),
        glm::vec3 const &                rightHandPosWorld = glm::vec3(0.f),
        glm::vec3 const &                rightHandVelWorld = glm::vec3(0.f));

} // namespace VCX::Labs::Animation
