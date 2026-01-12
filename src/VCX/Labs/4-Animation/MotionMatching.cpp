#include "Labs/4-Animation/MotionMatching.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include <glm/gtc/constants.hpp>

namespace VCX::Labs::Animation {
    namespace {
        glm::quat EulerToQuatLocal(std::vector<BVHChannel> const & chans, std::vector<float> const & frame, int start) {
            glm::quat q(1.f, 0.f, 0.f, 0.f);
            for (std::size_t i = 0; i < chans.size(); ++i) {
                auto ch = chans[i];
                if (ch == BVHChannel::Xrotation || ch == BVHChannel::Yrotation || ch == BVHChannel::Zrotation) {
                    float     rad = glm::radians(frame[start + int(i)]);
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
                auto  ch = chans[i];
                float v  = frame[start + int(i)];
                if (ch == BVHChannel::Xposition) t.x = v;
                if (ch == BVHChannel::Yposition) t.y = v;
                if (ch == BVHChannel::Zposition) t.z = v;
            }
            return t;
        }

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
            out.y         = 0.f;
            return out;
        }

        bool FootContactHeuristic(glm::vec3 const & pLocal, glm::vec3 const & vLocal) {
            // Typical thresholds; may tune per dataset.
            float heightTh   = 0.08f;
            float horizVelTh = 0.18f;
            float vertVelTh  = 0.60f;

            glm::vec2 hv(vLocal.x, vLocal.z);
            bool      low  = std::abs(pLocal.y) < heightTh;
            bool      slow = glm::dot(hv, hv) < horizVelTh * horizVelTh && std::abs(vLocal.y) < vertVelTh;
            return low && slow;
        }
    } // namespace

    LocalPose SampleLocalPose(BVHClip const & clip, float timeSeconds) {
        LocalPose pose;
        int       nJ = int(clip.Skeleton.Joints.size());
        pose.LocalRot.assign(nJ, glm::quat(1.f, 0.f, 0.f, 0.f));

        if (clip.FrameCount() == 0) return pose;
        float const T   = clip.FrameTime;
        float const dur = clip.FrameCount() * T;
        if (dur <= 0.f) return pose;

        float t = std::fmod(timeSeconds, dur);
        if (t < 0.f) t += dur;

        float f  = t / T;
        int   i0 = int(std::floor(f));
        int   i1 = std::min(i0 + 1, clip.FrameCount() - 1);
        float a  = f - float(i0);

        auto const & fr0 = clip.Frames[i0];
        auto const & fr1 = clip.Frames[i1];

        auto const & rootJ   = clip.Skeleton.Joints[clip.Skeleton.Root];
        glm::vec3    tr0     = ReadTranslation(rootJ.Channels, fr0, rootJ.ChannelStart);
        glm::vec3    tr1     = ReadTranslation(rootJ.Channels, fr1, rootJ.ChannelStart);
        pose.RootTranslation = glm::mix(tr0, tr1, a) + rootJ.Offset;

        for (int j = 0; j < nJ; ++j) {
            auto const & joint = clip.Skeleton.Joints[j];
            if (joint.Channels.empty()) {
                pose.LocalRot[j] = glm::quat(1.f, 0.f, 0.f, 0.f);
                continue;
            }
            glm::quat q0     = EulerToQuatLocal(joint.Channels, fr0, joint.ChannelStart);
            glm::quat q1     = EulerToQuatLocal(joint.Channels, fr1, joint.ChannelStart);
            pose.LocalRot[j] = glm::slerp(q0, q1, a);
        }

        return pose;
    }

    GlobalPose ComputeGlobalPose(BVHClip const & clip, LocalPose const & pose) {
        GlobalPose out;
        int        nJ = int(clip.Skeleton.Joints.size());
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
        Clip       = &clip;
        FrameTime  = clip.FrameTime;
        FrameCount = clip.FrameCount();
        Features.clear();
        RootPos.clear();
        RootYaw.clear();
        InvStd.fill(1.f);

        if (FrameCount <= 1) return;

        LeftFoot  = FindJointByName(clip.Skeleton, { "LeftFoot", "LeftAnkle", "lFoot", "LFoot" });
        RightFoot = FindJointByName(clip.Skeleton, { "RightFoot", "RightAnkle", "rFoot", "RFoot" });

        Head      = FindJointByName(clip.Skeleton, { "Head", "head" });
        LeftHand  = FindJointByName(clip.Skeleton, { "LeftHand", "LeftWrist", "lHand", "LHand" });
        RightHand = FindJointByName(clip.Skeleton, { "RightHand", "RightWrist", "rHand", "RHand" });

        auto sampleIndex = [&](int idx) { return (idx % FrameCount + FrameCount) % FrameCount; };

        Features.resize(FrameCount);
        RootPos.resize(FrameCount);
        RootYaw.resize(FrameCount);

        std::vector<glm::vec3> lfPos(FrameCount), rfPos(FrameCount);
        std::vector<glm::vec3> headPos(FrameCount), lhPos(FrameCount), rhPos(FrameCount);

        for (int i = 0; i < FrameCount; ++i) {
            LocalPose lp;
            lp.LocalRot.resize(int(clip.Skeleton.Joints.size()));
            auto const & fr    = clip.Frames[i];
            auto const & rootJ = clip.Skeleton.Joints[clip.Skeleton.Root];
            glm::vec3    tr    = ReadTranslation(rootJ.Channels, fr, rootJ.ChannelStart);
            lp.RootTranslation = tr + rootJ.Offset;

            for (int j = 0; j < int(clip.Skeleton.Joints.size()); ++j) {
                auto const & joint = clip.Skeleton.Joints[j];
                if (joint.Channels.empty()) lp.LocalRot[j] = glm::quat(1.f, 0.f, 0.f, 0.f);
                else lp.LocalRot[j] = EulerToQuatLocal(joint.Channels, fr, joint.ChannelStart);
            }

            auto gp = ComputeGlobalPose(clip, lp);

            RootPos[i] = gp.Pos[clip.Skeleton.Root];
            RootYaw[i] = ExtractYaw(gp.Rot[clip.Skeleton.Root]);

            lfPos[i]   = (LeftFoot >= 0) ? gp.Pos[LeftFoot] : RootPos[i];
            rfPos[i]   = (RightFoot >= 0) ? gp.Pos[RightFoot] : RootPos[i];
            headPos[i] = (Head >= 0) ? gp.Pos[Head] : RootPos[i];
            lhPos[i]   = (LeftHand >= 0) ? gp.Pos[LeftHand] : RootPos[i];
            rhPos[i]   = (RightHand >= 0) ? gp.Pos[RightHand] : RootPos[i];
        }

        std::array<float, 3> const futureT = { 0.2f, 0.4f, 0.6f };

        for (int i = 0; i < FrameCount; ++i) {
            int iNext = sampleIndex(i + 1);

            glm::vec3 rootVelWorld = (RootPos[iNext] - RootPos[i]) / FrameTime;
            glm::vec3 rootVelLocal = ToLocalXZ(RootYaw[i], rootVelWorld);

            Feature f {};
            int     k = 0;

            f[k++] = rootVelLocal.x;
            f[k++] = rootVelLocal.z;

            for (int t = 0; t < 3; ++t) {
                int       iF    = sampleIndex(i + int(std::round(futureT[t] / FrameTime)));
                glm::vec3 dp    = RootPos[iF] - RootPos[i];
                glm::vec3 local = ToLocalXZ(RootYaw[i], dp);
                f[k++]          = local.x;
                f[k++]          = local.z;
            }

            for (int t = 0; t < 3; ++t) {
                int       iF       = sampleIndex(i + int(std::round(futureT[t] / FrameTime)));
                glm::vec3 fwd      = RootYaw[iF] * glm::vec3(0.f, 0.f, 1.f);
                glm::vec3 localFwd = ToLocalXZ(RootYaw[i], fwd);
                glm::vec2 xz(localFwd.x, localFwd.z);
                if (glm::dot(xz, xz) < 1e-8f) xz = { 0.f, 1.f };
                xz     = glm::normalize(xz);
                f[k++] = xz.x;
                f[k++] = xz.y;
            }

            auto jointFeature = [&](glm::vec3 const & pos, glm::vec3 const & posNext, int & kk) {
                glm::quat inv    = glm::inverse(RootYaw[i]);
                glm::vec3 pLocal = inv * (pos - RootPos[i]);
                glm::vec3 vLocal = inv * ((posNext - pos) / FrameTime);
                f[kk++]          = pLocal.x;
                f[kk++]          = pLocal.y;
                f[kk++]          = pLocal.z;
                f[kk++]          = vLocal.x;
                f[kk++]          = vLocal.y;
                f[kk++]          = vLocal.z;
            };

            jointFeature(lfPos[i], lfPos[iNext], k);
            jointFeature(rfPos[i], rfPos[iNext], k);
            jointFeature(headPos[i], headPos[iNext], k);
            jointFeature(lhPos[i], lhPos[iNext], k);
            jointFeature(rhPos[i], rhPos[iNext], k);

            // contacts (L,R) in root-local
            {
                glm::quat inv = glm::inverse(RootYaw[i]);
                glm::vec3 pL  = inv * (lfPos[i] - RootPos[i]);
                glm::vec3 vL  = inv * ((lfPos[iNext] - lfPos[i]) / FrameTime);
                glm::vec3 pR  = inv * (rfPos[i] - RootPos[i]);
                glm::vec3 vR  = inv * ((rfPos[iNext] - rfPos[i]) / FrameTime);

                f[k++] = FootContactHeuristic(pL, vL) ? 1.f : 0.f;
                f[k++] = FootContactHeuristic(pR, vR) ? 1.f : 0.f;
            }

            Features[i] = f;
        }

        // Build InvStd for whitening (scale only; mean cancels in diffs).
        // Clamp to avoid exploding scales on low-variance dims.
        Feature mean {};
        mean.fill(0.f);
        for (auto const & f : Features)
            for (int d = 0; d < kFeatureDim; ++d)
                mean[d] += f[d];
        for (int d = 0; d < kFeatureDim; ++d)
            mean[d] /= float(FrameCount);

        Feature var {};
        var.fill(0.f);
        for (auto const & f : Features)
            for (int d = 0; d < kFeatureDim; ++d) {
                float x = f[d] - mean[d];
                var[d] += x * x;
            }
        for (int d = 0; d < kFeatureDim; ++d) {
            var[d] /= float(FrameCount);
            float sigma = std::sqrt(std::max(var[d], 1e-8f));
            // contact dims: don't whiten too aggressively
            if (d >= kFeatureDim - 2) sigma = std::max(sigma, 0.35f);
            InvStd[d] = 1.f / std::max(sigma, 1e-3f);
        }
    }

    MatchResult MotionDatabase::FindBestMatch(
        Feature const & query,
        Feature const & weights,
        int             currentIndex,
        int             excludeWindow) const {
        MatchResult best { .Index = currentIndex, .Cost = std::numeric_limits<float>::infinity() };
        if (Features.empty()) return best;

        int stride = std::max(1, SearchStride);

        auto evalCost = [&](int i) -> float {
            float cost = 0.f;
            for (int d = 0; d < kFeatureDim; ++d) {
                float w = weights[d];
                if (w == 0.f) continue;
                float diff = (Features[i][d] - query[d]) * InvStd[d];
                cost += w * diff * diff;
            }
            return cost;
        };

        // coarse pass
        for (int i = 0; i < FrameCount; i += stride) {
            if (excludeWindow > 0 && std::abs(i - currentIndex) <= excludeWindow) continue;
            float cost = evalCost(i);
            if (cost < best.Cost) {
                best.Index = i;
                best.Cost  = cost;
            }
        }

        // refine around coarse best
        int start = std::max(0, best.Index - stride);
        int end   = std::min(FrameCount - 1, best.Index + stride);
        for (int i = start; i <= end; ++i) {
            if (excludeWindow > 0 && std::abs(i - currentIndex) <= excludeWindow) continue;
            float cost = evalCost(i);
            if (cost < best.Cost) {
                best.Index = i;
                best.Cost  = cost;
            }
        }

        return best;
    }

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
        glm::vec3 const &                headPosWorld,
        glm::vec3 const &                headVelWorld,
        glm::vec3 const &                leftHandPosWorld,
        glm::vec3 const &                leftHandVelWorld,
        glm::vec3 const &                rightHandPosWorld,
        glm::vec3 const &                rightHandVelWorld) {
        Feature   q {};
        int       k      = 0;
        glm::quat invYaw = glm::inverse(rootYawWorld);

        glm::vec3 velLocal = invYaw * rootVelWorld;
        velLocal.y         = 0.f;
        q[k++]             = velLocal.x;
        q[k++]             = velLocal.z;

        for (int i = 0; i < 3; ++i) {
            glm::vec3 dp    = desiredFuturePosWorld[i] - rootPosWorld;
            glm::vec3 local = invYaw * dp;
            local.y         = 0.f;
            q[k++]          = local.x;
            q[k++]          = local.z;
        }

        for (int i = 0; i < 3; ++i) {
            glm::vec3 fwd = desiredFutureFwdWorld[i];
            fwd.y         = 0.f;
            if (glm::dot(fwd, fwd) < 1e-8f) fwd = { 0.f, 0.f, 1.f };
            fwd             = glm::normalize(fwd);
            glm::vec3 local = invYaw * fwd;
            local.y         = 0.f;
            glm::vec2 xz(local.x, local.z);
            if (glm::dot(xz, xz) < 1e-8f) xz = { 0.f, 1.f };
            xz     = glm::normalize(xz);
            q[k++] = xz.x;
            q[k++] = xz.y;
        }

        auto joint = [&](glm::vec3 const & pW, glm::vec3 const & vW) {
            glm::vec3 pL = invYaw * (pW - rootPosWorld);
            glm::vec3 vL = invYaw * vW;
            q[k++]       = pL.x;
            q[k++]       = pL.y;
            q[k++]       = pL.z;
            q[k++]       = vL.x;
            q[k++]       = vL.y;
            q[k++]       = vL.z;
        };

        joint(leftFootPosWorld, leftFootVelWorld);
        joint(rightFootPosWorld, rightFootVelWorld);
        joint(headPosWorld, headVelWorld);
        joint(leftHandPosWorld, leftHandVelWorld);
        joint(rightHandPosWorld, rightHandVelWorld);

        // contacts (L,R)
        {
            glm::vec3 pL = invYaw * (leftFootPosWorld - rootPosWorld);
            glm::vec3 vL = invYaw * leftFootVelWorld;
            glm::vec3 pR = invYaw * (rightFootPosWorld - rootPosWorld);
            glm::vec3 vR = invYaw * rightFootVelWorld;

            q[k++] = FootContactHeuristic(pL, vL) ? 1.f : 0.f;
            q[k++] = FootContactHeuristic(pR, vR) ? 1.f : 0.f;
        }

        return q;
    }
} // namespace VCX::Labs::Animation
