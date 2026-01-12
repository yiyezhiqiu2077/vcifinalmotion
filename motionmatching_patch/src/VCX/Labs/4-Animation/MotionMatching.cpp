#include "Labs/4-Animation/MotionMatching.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include <glm/gtc/constants.hpp>

namespace VCX::Labs::Animation {
    namespace {
        glm::quat EulerToQuatLocal(std::vector<BVHChannel> const & chans, std::vector<float> const & frame, int start) {
            // BVH channels order defines local Euler order.
            glm::quat q(1.f, 0.f, 0.f, 0.f);
            for (std::size_t i = 0; i < chans.size(); ++i) {
                auto ch = chans[i];
                if (ch == BVHChannel::Xrotation || ch == BVHChannel::Yrotation || ch == BVHChannel::Zrotation) {
                    float rad = glm::radians(frame[start + int(i)]);
                    glm::vec3 axis(0.f);
                    if (ch == BVHChannel::Xrotation) axis = { 1.f, 0.f, 0.f };
                    if (ch == BVHChannel::Yrotation) axis = { 0.f, 1.f, 0.f };
                    if (ch == BVHChannel::Zrotation) axis = { 0.f, 0.f, 1.f };
                    q = q * glm::angleAxis(rad, axis);
                }
            }
            return q;
        }

        glm::vec3 ReadTranslation(std::vector<BVHChannel> const & chans, std::vector<float> const & frame, int start) {
            glm::vec3 t(0.f);
            for (std::size_t i = 0; i < chans.size(); ++i) {
                auto ch = chans[i];
                float v = frame[start + int(i)];
                if (ch == BVHChannel::Xposition) t.x = v;
                if (ch == BVHChannel::Yposition) t.y = v;
                if (ch == BVHChannel::Zposition) t.z = v;
            }
            return t;
        }

        // Heuristic to find foot joints by name.
        int FindJointByName(BVHSkeleton const & skel, std::initializer_list<std::string_view> const hints) {
            for (auto const & h : hints) {
                for (int i = 0; i < int(skel.Joints.size()); ++i) {
                    auto const & n = skel.Joints[i].Name;
                    if (n.find(std::string(h)) != std::string::npos)
                        return i;
                }
            }
            return -1;
        }

        glm::vec3 ToLocalXZ(glm::quat const & yaw, glm::vec3 const & v) {
            glm::vec3 out = glm::inverse(yaw) * v;
            out.y = 0.f;
            return out;
        }
    }

    LocalPose SampleLocalPose(BVHClip const & clip, float timeSeconds) {
        LocalPose pose;
        int nJ = int(clip.Skeleton.Joints.size());
        pose.LocalRot.assign(nJ, glm::quat(1.f, 0.f, 0.f, 0.f));

        if (clip.FrameCount() == 0) return pose;
        float const T = clip.FrameTime;
        float const dur = clip.FrameCount() * T;
        if (dur <= 0.f) return pose;

        // Wrap time for looping.
        float t = std::fmod(timeSeconds, dur);
        if (t < 0.f) t += dur;

        float f = t / T;
        int i0 = int(std::floor(f));
        int i1 = std::min(i0 + 1, clip.FrameCount() - 1);
        float a = f - float(i0);

        auto const & fr0 = clip.Frames[i0];
        auto const & fr1 = clip.Frames[i1];

        // Root translation uses channels (if present) + root offset.
        auto const & rootJ = clip.Skeleton.Joints[clip.Skeleton.Root];
        glm::vec3 tr0 = ReadTranslation(rootJ.Channels, fr0, rootJ.ChannelStart);
        glm::vec3 tr1 = ReadTranslation(rootJ.Channels, fr1, rootJ.ChannelStart);
        pose.RootTranslation = glm::mix(tr0, tr1, a) + rootJ.Offset;

        for (int j = 0; j < nJ; ++j) {
            auto const & joint = clip.Skeleton.Joints[j];
            if (joint.Channels.empty()) {
                pose.LocalRot[j] = glm::quat(1.f, 0.f, 0.f, 0.f);
                continue;
            }
            glm::quat q0 = EulerToQuatLocal(joint.Channels, fr0, joint.ChannelStart);
            glm::quat q1 = EulerToQuatLocal(joint.Channels, fr1, joint.ChannelStart);
            pose.LocalRot[j] = glm::slerp(q0, q1, a);
        }

        return pose;
    }

    GlobalPose ComputeGlobalPose(BVHClip const & clip, LocalPose const & pose) {
        GlobalPose out;
        int nJ = int(clip.Skeleton.Joints.size());
        out.Pos.assign(nJ, glm::vec3(0.f));
        out.Rot.assign(nJ, glm::quat(1.f, 0.f, 0.f, 0.f));

        for (int j = 0; j < nJ; ++j) {
            auto const & joint = clip.Skeleton.Joints[j];
            if (joint.Parent < 0) {
                out.Pos[j] = pose.RootTranslation;
                out.Rot[j] = pose.LocalRot[j];
            } else {
                out.Pos[j] = out.Pos[joint.Parent] + out.Rot[joint.Parent] * joint.Offset;
                out.Rot[j] = out.Rot[joint.Parent] * pose.LocalRot[j];
            }
        }

        return out;
    }

    void MotionDatabase::Build(BVHClip const & clip) {
        Clip = &clip;
        FrameTime = clip.FrameTime;
        FrameCount = clip.FrameCount();
        Features.clear();
        RootPos.clear();
        RootYaw.clear();

        if (FrameCount <= 1) return;

        // Foot joints
        LeftFoot = FindJointByName(clip.Skeleton, {"LeftFoot", "LeftAnkle", "lFoot", "LFoot", "Left"});
        RightFoot = FindJointByName(clip.Skeleton, {"RightFoot", "RightAnkle", "rFoot", "RFoot", "Right"});
        // Fallback: try deepest leaves under root's children.
        auto fallbackLeaf = [&](int side) -> int {
            // side: 0 left, 1 right
            int best = -1;
            int bestDepth = -1;
            std::function<void(int,int)> dfs = [&](int j, int d) {
                auto const & children = clip.Skeleton.Joints[j].Children;
                if (children.empty()) {
                    if (d > bestDepth) { bestDepth = d; best = j; }
                }
                for (int c : children) dfs(c, d + 1);
            };
            // Try each child of root.
            auto const & rootChildren = clip.Skeleton.Joints[clip.Skeleton.Root].Children;
            if (rootChildren.empty()) return -1;
            int start = side < int(rootChildren.size()) ? rootChildren[side] : rootChildren.front();
            dfs(start, 0);
            return best;
        };
        if (LeftFoot < 0) LeftFoot = fallbackLeaf(0);
        if (RightFoot < 0) RightFoot = fallbackLeaf(1);

        Features.resize(FrameCount);
        RootPos.resize(FrameCount);
        RootYaw.resize(FrameCount);

        // Precompute global poses per frame (for foot features).
        std::vector<glm::vec3> lfPos(FrameCount), rfPos(FrameCount);

        for (int i = 0; i < FrameCount; ++i) {
            LocalPose lp;
            lp.LocalRot.resize(int(clip.Skeleton.Joints.size()));
            // No interpolation: sample exact frame.
            auto const & fr = clip.Frames[i];
            auto const & rootJ = clip.Skeleton.Joints[clip.Skeleton.Root];
            glm::vec3 tr = ReadTranslation(rootJ.Channels, fr, rootJ.ChannelStart);
            lp.RootTranslation = tr + rootJ.Offset;
            for (int j = 0; j < int(clip.Skeleton.Joints.size()); ++j) {
                auto const & joint = clip.Skeleton.Joints[j];
                if (joint.Channels.empty()) {
                    lp.LocalRot[j] = glm::quat(1.f, 0.f, 0.f, 0.f);
                } else {
                    lp.LocalRot[j] = EulerToQuatLocal(joint.Channels, fr, joint.ChannelStart);
                }
            }
            auto gp = ComputeGlobalPose(clip, lp);
            RootPos[i] = gp.Pos[clip.Skeleton.Root];
            RootYaw[i] = ExtractYaw(gp.Rot[clip.Skeleton.Root]);
            lfPos[i] = (LeftFoot >= 0) ? gp.Pos[LeftFoot] : RootPos[i];
            rfPos[i] = (RightFoot >= 0) ? gp.Pos[RightFoot] : RootPos[i];
        }

        auto sampleIndex = [&](int idx) { return (idx % FrameCount + FrameCount) % FrameCount; };

        // Feature times for trajectory (seconds)
        std::array<float, 3> const futureT = { 0.2f, 0.4f, 0.6f };

        for (int i = 0; i < FrameCount; ++i) {
            int iNext = sampleIndex(i + 1);
            glm::vec3 rootDelta = (RootPos[iNext] - RootPos[i]) / FrameTime;
            glm::vec3 rootVelLocal = ToLocalXZ(RootYaw[i], rootDelta);

            Feature f{};
            int k = 0;
            // Root vel xz
            f[k++] = rootVelLocal.x;
            f[k++] = rootVelLocal.z;

            // Future trajectory positions xz
            for (int t = 0; t < 3; ++t) {
                int iF = sampleIndex(i + int(std::round(futureT[t] / FrameTime)));
                glm::vec3 dp = RootPos[iF] - RootPos[i];
                glm::vec3 local = ToLocalXZ(RootYaw[i], dp);
                f[k++] = local.x;
                f[k++] = local.z;
            }

            // Future facing directions xz
            for (int t = 0; t < 3; ++t) {
                int iF = sampleIndex(i + int(std::round(futureT[t] / FrameTime)));
                glm::vec3 fwd = RootYaw[iF] * glm::vec3(0.f, 0.f, 1.f);
                glm::vec3 localFwd = ToLocalXZ(RootYaw[i], fwd);
                // normalize on XZ plane
                glm::vec2 xz(localFwd.x, localFwd.z);
                if (glm::length2(xz) < 1e-8f) xz = {0.f, 1.f};
                xz = glm::normalize(xz);
                f[k++] = xz.x;
                f[k++] = xz.y;
            }

            // Feet pos/vel (local)
            auto footFeature = [&](glm::vec3 const & pos, glm::vec3 const & posNext, int & kk) {
                glm::vec3 pLocal = glm::inverse(RootYaw[i]) * (pos - RootPos[i]);
                glm::vec3 vLocal = glm::inverse(RootYaw[i]) * ((posNext - pos) / FrameTime);
                f[kk++] = pLocal.x; f[kk++] = pLocal.y; f[kk++] = pLocal.z;
                f[kk++] = vLocal.x; f[kk++] = vLocal.y; f[kk++] = vLocal.z;
            };

            int lfN = sampleIndex(i + 1);
            footFeature(lfPos[i], lfPos[lfN], k);
            int rfN = sampleIndex(i + 1);
            footFeature(rfPos[i], rfPos[rfN], k);

            Features[i] = f;
        }
    }

    MatchResult MotionDatabase::FindBestMatch(
        Feature const & query,
        Feature const & weights,
        int currentIndex,
        int excludeWindow) const {
        MatchResult best{ .Index = currentIndex, .Cost = std::numeric_limits<float>::infinity() };
        if (Features.empty()) return best;

        for (int i = 0; i < FrameCount; ++i) {
            if (excludeWindow > 0 && std::abs(i - currentIndex) <= excludeWindow) continue;
            float cost = 0.f;
            for (int d = 0; d < kFeatureDim; ++d) {
                float w = weights[d];
                if (w == 0.f) continue;
                float diff = Features[i][d] - query[d];
                cost += w * diff * diff;
            }
            if (cost < best.Cost) {
                best.Index = i;
                best.Cost = cost;
            }
        }
        return best;
    }

    Feature BuildQueryFeature(
        glm::vec3 const & rootPosWorld,
        glm::quat const & rootYawWorld,
        glm::vec3 const & rootVelWorld,
        std::array<glm::vec3, 3> const & desiredFuturePosWorld,
        std::array<glm::vec3, 3> const & desiredFutureFwdWorld,
        glm::vec3 const & leftFootPosWorld,
        glm::vec3 const & leftFootVelWorld,
        glm::vec3 const & rightFootPosWorld,
        glm::vec3 const & rightFootVelWorld) {

        Feature q{};
        int k = 0;
        glm::quat invYaw = glm::inverse(rootYawWorld);

        glm::vec3 velLocal = invYaw * rootVelWorld;
        velLocal.y = 0.f;
        q[k++] = velLocal.x;
        q[k++] = velLocal.z;

        for (int i = 0; i < 3; ++i) {
            glm::vec3 dp = desiredFuturePosWorld[i] - rootPosWorld;
            glm::vec3 local = invYaw * dp;
            local.y = 0.f;
            q[k++] = local.x;
            q[k++] = local.z;
        }

        for (int i = 0; i < 3; ++i) {
            glm::vec3 fwd = desiredFutureFwdWorld[i];
            fwd.y = 0.f;
            if (glm::length2(fwd) < 1e-8f) fwd = {0.f, 0.f, 1.f};
            fwd = glm::normalize(fwd);
            glm::vec3 local = invYaw * fwd;
            local.y = 0.f;
            glm::vec2 xz(local.x, local.z);
            if (glm::length2(xz) < 1e-8f) xz = {0.f, 1.f};
            xz = glm::normalize(xz);
            q[k++] = xz.x;
            q[k++] = xz.y;
        }

        auto foot = [&](glm::vec3 const & pW, glm::vec3 const & vW) {
            glm::vec3 pL = invYaw * (pW - rootPosWorld);
            glm::vec3 vL = invYaw * vW;
            q[k++] = pL.x; q[k++] = pL.y; q[k++] = pL.z;
            q[k++] = vL.x; q[k++] = vL.y; q[k++] = vL.z;
        };

        foot(leftFootPosWorld, leftFootVelWorld);
        foot(rightFootPosWorld, rightFootVelWorld);

        return q;
    }
}
