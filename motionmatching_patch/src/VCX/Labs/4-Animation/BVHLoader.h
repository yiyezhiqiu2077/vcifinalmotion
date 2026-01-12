#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace VCX::Labs::Animation {

    enum class BVHChannel : std::uint8_t {
        Xposition,
        Yposition,
        Zposition,
        Xrotation,
        Yrotation,
        Zrotation,
    };

    struct BVHJoint {
        std::string            Name;
        int                    Parent { -1 };
        glm::vec3              Offset { 0.f };
        std::vector<int>       Children;
        std::vector<BVHChannel> Channels;
        int                    ChannelStart { 0 }; // start index in a motion frame
    };

    struct BVHSkeleton {
        std::vector<BVHJoint> Joints;
        int                   Root { 0 };
    };

    struct BVHClip {
        BVHSkeleton                     Skeleton;
        float                           FrameTime { 1.f / 60.f };
        int                             TotalChannels { 0 };
        std::vector<std::vector<float>> Frames; // Frames[frameIndex][channelIndex]

        int FrameCount() const { return int(Frames.size()); }
    };

    // Throws std::runtime_error on parse errors.
    BVHClip LoadBVH(std::filesystem::path const & path);

    // Utility: extract yaw-only rotation (rotation around +Y) from an orientation.
    glm::quat ExtractYaw(glm::quat const & q);

} // namespace VCX::Labs::Animation
