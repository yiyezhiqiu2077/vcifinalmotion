
#include "Labs/4-Animation/CaseMotionMatching.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>

#include <GLFW/glfw3.h> // ✅ 用 GLFW 读键盘，避免 ImGui 焦点问题

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "Engine/app.h"
#include "Labs/Common/ImGuiHelper.h" 



namespace VCX::Labs::Animation {
    namespace {
        glm::quat YawFromAngle(float yaw) {
            return glm::angleAxis(yaw, glm::vec3(0.f, 1.f, 0.f));
        }

        float WrapAngle(float a) {
            while (a > glm::pi<float>()) a -= glm::two_pi<float>();
            while (a < -glm::pi<float>()) a += glm::two_pi<float>();
            return a;
        }

        glm::quat SafeSlerp(glm::quat const & a, glm::quat const & b, float t) {
            return glm::normalize(glm::slerp(a, b, t));
        }

        // yaw-only quaternion -> yaw angle
        float YawAngleFromYawQuat(glm::quat const & qYaw) {
            // for yaw-only quat: q = (w, 0, y, 0)
            return 2.f * std::atan2(qYaw.y, qYaw.w);
        }

        void AppendBox(
            std::vector<CaseMotionMatching::SkinnedVertex> & verts,
            std::vector<std::uint32_t> &                     indices,
            int                                              boneIndex,
            glm::mat3 const &                                R,
            float                                            length,
            float                                            r) {
            struct Face {
                glm::vec3 n;
                glm::vec3 v0, v1, v2, v3;
            };
            float               z0    = 0.f;
            float               z1    = length;
            std::array<Face, 6> faces = {
                Face {  { 1, 0, 0 },  { r, -r, z0 },  { r, r, z0 },  { r, r, z1 },  { r, -r, z1 } },
                Face { { -1, 0, 0 }, { -r, -r, z1 }, { -r, r, z1 }, { -r, r, z0 }, { -r, -r, z0 } },
                Face {  { 0, 1, 0 },  { -r, r, z0 },  { r, r, z0 },  { r, r, z1 },  { -r, r, z1 } },
                Face { { 0, -1, 0 }, { -r, -r, z1 }, { r, -r, z1 }, { r, -r, z0 }, { -r, -r, z0 } },
                Face {  { 0, 0, 1 }, { -r, -r, z1 }, { -r, r, z1 },  { r, r, z1 },  { r, -r, z1 } },
                Face { { 0, 0, -1 }, { -r, -r, z0 }, { r, -r, z0 },  { r, r, z0 },  { -r, r, z0 } },
            };

            glm::vec4 bone(float(boneIndex), 0.f, 0.f, 0.f);
            glm::vec4 w(1.f, 0.f, 0.f, 0.f);

            for (auto const & f : faces) {
                std::uint32_t base = std::uint32_t(verts.size());
                glm::vec3     n    = glm::normalize(R * f.n);
                auto          addV = [&](glm::vec3 const & p) {
                    verts.push_back({
                                 .Position = R * p,
                                 .Normal   = n,
                                 .Bone     = bone,
                                 .Weight   = w,
                    });
                };
                addV(f.v0);
                addV(f.v1);
                addV(f.v2);
                addV(f.v3);
                indices.push_back(base + 0);
                indices.push_back(base + 1);
                indices.push_back(base + 2);
                indices.push_back(base + 0);
                indices.push_back(base + 2);
                indices.push_back(base + 3);
            }
        }

        glm::quat RotationBetweenVectors(glm::vec3 from, glm::vec3 to) {
            float fromLen2 = glm::dot(from, from);
            float toLen2   = glm::dot(to, to);
            if (fromLen2 < 1e-8f || toLen2 < 1e-8f)
                return glm::quat(1.f, 0.f, 0.f, 0.f);

            from = glm::normalize(from);
            to   = glm::normalize(to);

            float cosTheta = glm::dot(from, to);
            if (cosTheta > 1.f - 1e-6f)
                return glm::quat(1.f, 0.f, 0.f, 0.f);

            if (cosTheta < -1.f + 1e-6f) {
                glm::vec3 axis = glm::cross(glm::vec3(1.f, 0.f, 0.f), from);
                if (glm::dot(axis, axis) < 1e-6f)
                    axis = glm::cross(glm::vec3(0.f, 1.f, 0.f), from);
                axis = glm::normalize(axis);
                return glm::angleAxis(glm::pi<float>(), axis);
            }

            glm::vec3 axis = glm::cross(from, to);
            float     s    = std::sqrt((1.f + cosTheta) * 2.f);
            float     invs = 1.f / s;
            return glm::quat(s * 0.5f, axis.x * invs, axis.y * invs, axis.z * invs);
        }

        glm::mat3 RotationFromTo(glm::vec3 from, glm::vec3 to) {
            if (glm::dot(from, from) < 1e-8f || glm::dot(to, to) < 1e-8f)
                return glm::mat3(1.f);
            return glm::mat3_cast(RotationBetweenVectors(from, to));
        }
    } // namespace

    CaseMotionMatching::CaseMotionMatching():
        _flatProgram(Engine::GL::UniqueProgram({
            Engine::GL::SharedShader("assets/shaders/flat.vert"),
            Engine::GL::SharedShader("assets/shaders/flat.frag"),
        })),
        _skinnedProgram(Engine::GL::UniqueProgram({
            Engine::GL::SharedShader("assets/shaders/skinned.vert"),
            Engine::GL::SharedShader("assets/shaders/skinned.frag"),
        })),
        _skeletonLines(
            Engine::GL::VertexLayout()
                .Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines),
        _trajectoryLines(
            Engine::GL::VertexLayout()
                .Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines),
        _bonesUbo(kBonesBindingPoint, Engine::GL::DrawFrequency::Stream),
        _mesh(
            Engine::GL::VertexLayout()
                .Add<SkinnedVertex>("vtx", Engine::GL::DrawFrequency::Static)
                .At(0, &SkinnedVertex::Position)
                .At(1, &SkinnedVertex::Normal)
                .At(2, &SkinnedVertex::Bone)
                .At(3, &SkinnedVertex::Weight),
            Engine::GL::PrimitiveType::Triangles) {
        _weights.fill(1.f);

        // Root vel
        _weights[0] = 2.f;
        _weights[1] = 2.f;
        // Traj pos (6)
        for (int i = 2; i < 8; ++i) _weights[i] = 3.f;
        // Facing (6)
        for (int i = 8; i < 14; ++i) _weights[i] = 1.5f;
        // Feet (12)
        for (int i = 14; i < 26; ++i) _weights[i] = 1.0f;
        // Upper body (18)
        for (int i = 26; i < 44; ++i) _weights[i] = 0.8f;
        // Contact (2)
        _weights[44] = 6.f;
        _weights[45] = 6.f;

        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

        _skinnedProgram.BindUniformBlock("Bones", kBonesBindingPoint);

        for (auto & m : _boneMatrices) m = glm::mat4(1.0f);
        for (auto & m : _bonesBlock.u_Bones) m = glm::mat4(1.0f);
        _bonesUbo.Update(_bonesBlock);

        std::snprintf(_bvhPathBuf.data(), _bvhPathBuf.size(), "%s", "assets/mocap/mm_synthetic_walk_turn.bvh");
        loadBVH(std::filesystem::path(_bvhPathBuf.data()));
    }

    void CaseMotionMatching::loadBVH(std::filesystem::path const & path) {
        try {
            _clip       = LoadBVH(path);
            _clipLoaded = true;

            rebuildDatabase();
            buildProceduralMesh();

            _align        = Align {};
            _dbTime       = 0.f;
            _currentIndex = 0;
            _lastMatch    = { 0, 0.f };

            _hasPrevKinematics = false;
            _leftLocked = _rightLocked = false;

            auto pose0 = sampleAligned(0.f, _align);
            if (! pose0.Pos.empty()) {
                fitCameraToPose(pose0);

                // ✅ 初始化 controller yaw = 当前动画 root yaw
                glm::quat yawQ = ExtractYaw(pose0.Rot[_clip.Skeleton.Root]);
                _desiredYaw    = YawAngleFromYawQuat(yawQ);

                // ✅ 初始化 prev kinematics，避免第一帧 query 乱跳
                int root          = _clip.Skeleton.Root;
                _prevRootPosWorld = pose0.Pos[root];

                auto safePos = [&](int idx) -> glm::vec3 {
                    if (idx >= 0 && idx < int(pose0.Pos.size())) return pose0.Pos[idx];
                    return pose0.Pos[root];
                };
                _prevLeftFootPosWorld  = safePos(_db.LeftFoot);
                _prevRightFootPosWorld = safePos(_db.RightFoot);
                _prevHeadPosWorld      = safePos(_db.Head);
                _prevLeftHandPosWorld  = safePos(_db.LeftHand);
                _prevRightHandPosWorld = safePos(_db.RightHand);
                _hasPrevKinematics     = true;
            }

            resetFollowRootState();
        } catch (...) {
            _clipLoaded = false;
        }
    }

    void CaseMotionMatching::rebuildDatabase() {
        if (! _clipLoaded) return;
        _db.Build(_clip);
        _db.SearchStride = std::clamp(_db.SearchStride, 1, 16);
    }

    void CaseMotionMatching::buildProceduralMesh() {
        _meshVertices.clear();
        _meshIndices.clear();
        if (! _clipLoaded) return;

        // Estimate scale from bone lengths
        float avgLen = 0.0f;
        int   cnt    = 0;
        for (int j = 0; j < int(_clip.Skeleton.Joints.size()) && j < int(kMaxBones); ++j) {
            auto const & joint = _clip.Skeleton.Joints[j];
            for (int c : joint.Children) {
                float len = glm::length(_clip.Skeleton.Joints[c].Offset);
                if (len > 1e-4f) {
                    avgLen += len;
                    cnt++;
                }
            }
        }
        avgLen = (cnt > 0) ? (avgLen / float(cnt)) : 1.0f;

        // ✅ 更细一点的半径，避免像柱子
        float const radius = std::max(0.008f * avgLen, 0.02f);

        // ✅ 关键修正：对每个 joint 的每个 child 都生成一段
        for (int j = 0; j < int(_clip.Skeleton.Joints.size()) && j < int(kMaxBones); ++j) {
            auto const & joint = _clip.Skeleton.Joints[j];
            for (int child : joint.Children) {
                glm::vec3 off = _clip.Skeleton.Joints[child].Offset;
                float     len = glm::length(off);
                if (len < 1e-4f) continue;
                glm::vec3 dir = off / len;
                glm::mat3 R   = RotationFromTo(glm::vec3(0.f, 0.f, 1.f), dir);
                AppendBox(_meshVertices, _meshIndices, j, R, len, radius);
            }
        }

        _mesh.UpdateVertexBuffer(
            "vtx",
            Engine::make_span_bytes(std::span<SkinnedVertex const>(_meshVertices.data(), _meshVertices.size())));
        _mesh.UpdateElementBuffer(std::span<std::uint32_t const>(_meshIndices.data(), _meshIndices.size()));
    }

    void CaseMotionMatching::OnSetupPropsUI() {
        ImGui::TextWrapped("W/S: forward/back   A/D: turn   Shift: run");
        ImGui::TextWrapped("Mouse: orbit camera (drag) / wheel zoom (if enabled).");

        if (ImGui::CollapsingHeader("BVH", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputText("BVH Path", _bvhPathBuf.data(), _bvhPathBuf.size());
            if (ImGui::Button("Reload BVH")) _reloadRequested = true;
            ImGui::SameLine();
            ImGui::TextDisabled(_clipLoaded ? "(loaded)" : "(load failed)");
            ImGui::Text("Frames: %d  dt: %.4f", _clipLoaded ? _clip.FrameCount() : 0, _clipLoaded ? _clip.FrameTime : 0.f);
        }

        if (ImGui::CollapsingHeader("Runtime", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button(_paused ? "Resume" : "Pause")) _paused = ! _paused;

            ImGui::Checkbox("Mesh", &_showMesh);
            ImGui::SameLine();
            ImGui::Checkbox("Skeleton", &_showSkeleton);
            ImGui::SameLine();
            ImGui::Checkbox("Trajectory", &_showTrajectory);

            ImGui::Checkbox("Follow Root", &_followRoot);
            ImGui::SameLine();
            if (ImGui::Button("Fit Camera") && _clipLoaded) {
                auto poseNow = sampleAligned(_dbTime, _align);
                fitCameraToPose(poseNow);
            }

            ImGui::SliderFloat("Walk Speed", &_walkSpeed, 0.2f, 3.0f);
            ImGui::SliderFloat("Run Speed", &_runSpeed, 0.5f, 6.0f);
            ImGui::SliderFloat("Turn Speed (deg/s)", &_turnSpeedDeg, 30.f, 300.f);

            ImGui::SliderFloat("Match Interval (s)", &_searchInterval, 0.02f, 0.30f);
            ImGui::SliderInt("Exclude Window (frames)", &_excludeWindow, 0, 60);
            ImGui::SliderInt("Search Stride", &_db.SearchStride, 1, 16);

            ImGui::Checkbox("Use Inertialization", &_useInertialization);
            ImGui::SliderFloat("Inertial HalfLife (s)", &_inertialHalfLife, 0.03f, 0.30f);

            ImGui::Checkbox("Enable Foot Lock", &_enableFootLock);
            ImGui::SliderFloat("FootLock VelTh", &_footLockVelThresh, 0.05f, 0.50f);
            ImGui::SliderFloat("FootLock HeightTh", &_footLockHeightThresh, 0.02f, 0.20f);

            ImGui::Text("Current frame: %d", _currentIndex);
            ImGui::Text("Last match: #%d  cost: %.4f", _lastMatch.Index, _lastMatch.Cost);

            ImGui::Text("Root pos (world): %.2f %.2f %.2f", _prevRootPosWorld.x, _prevRootPosWorld.y, _prevRootPosWorld.z);
        }
    }

    void CaseMotionMatching::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);
    }

    GlobalPose CaseMotionMatching::sampleAligned(float dbTime, Align const & align) const {
        auto lp = SampleLocalPose(_clip, dbTime);
        auto gp = ComputeGlobalPose(_clip, lp);

        GlobalPose out;
        int        n = int(gp.Pos.size());
        out.Pos.resize(n);
        out.Rot.resize(n);
        for (int i = 0; i < n; ++i) {
            out.Pos[i] = align.Pos + align.Rot * gp.Pos[i];
            out.Rot[i] = align.Rot * gp.Rot[i];
        }
        return out;
    }

    void CaseMotionMatching::fitCameraToPose(GlobalPose const & poseWorld) {
        if (poseWorld.Pos.empty()) return;

        glm::vec3 mn(std::numeric_limits<float>::infinity());
        glm::vec3 mx(-std::numeric_limits<float>::infinity());
        for (auto const & p : poseWorld.Pos) {
            mn = glm::min(mn, p);
            mx = glm::max(mx, p);
        }
        glm::vec3 center = 0.5f * (mn + mx);
        float     radius = 0.5f * glm::length(mx - mn);
        radius           = std::max(radius, 1.0f);

        _camera.Target = center;
        _camera.Eye    = center + glm::vec3(-1.6f * radius, 0.9f * radius, 1.6f * radius);
        _camera.ZNear  = 0.01f;
        _camera.ZFar   = std::max(200.f, 12.f * radius);

        _cameraManager.Save(_camera);
        resetFollowRootState();
    }

    void CaseMotionMatching::updateController(float dt) {
        GLFWwindow * win  = glfwGetCurrentWindow();
        auto         down = [&](int key) -> bool {
            return win && glfwGetKey(win, key) == GLFW_PRESS;
        };

        // ✅ WASD + Shift
        bool W     = down(GLFW_KEY_W);
        bool S     = down(GLFW_KEY_S);
        bool A     = down(GLFW_KEY_A);
        bool D     = down(GLFW_KEY_D);
        bool shift = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);

        // 转向：A 左转，D 右转
        float yawRate = glm::radians(_turnSpeedDeg);
        if (A) _desiredYaw += yawRate * dt;
        if (D) _desiredYaw -= yawRate * dt;

        // 速度：W 前进，S 后退（后退给个折扣更像游戏）
        float base  = shift ? _runSpeed : _walkSpeed;
        float speed = 0.f;
        if (W) speed += base;
        if (S) speed -= 0.6f * base;

        glm::quat yawQ = glm::angleAxis(_desiredYaw, glm::vec3(0.f, 1.f, 0.f));
        glm::vec3 fwd  = yawQ * glm::vec3(0.f, 0.f, 1.f);

        _desiredVelWorld = fwd * speed;

        // 未来轨迹（3 个时间点）
        std::array<float, 3> const ts      = { 0.2f, 0.4f, 0.6f };
        glm::vec3                  basePos = _hasPrevKinematics ? _prevRootPosWorld : glm::vec3(0.f);

        for (int i = 0; i < 3; ++i) {
            _desiredFuturePosWorld[i] = basePos + _desiredVelWorld * ts[i];
            _desiredFutureFwdWorld[i] = fwd;
        }
    }


  

    void CaseMotionMatching::updateMatching(float dt) {
        if (! _clipLoaded || _db.Features.empty()) return;

        _searchTimer += dt;
        if (_searchTimer < _searchInterval) return;
        _searchTimer = 0.f;

        auto      poseNow = sampleAligned(_dbTime, _align);
        glm::vec3 rootPos = poseNow.Pos[_clip.Skeleton.Root];
        glm::quat rootYaw = ExtractYaw(poseNow.Rot[_clip.Skeleton.Root]);

        glm::vec3 rootVel = (_hasPrevKinematics && dt > 0.f) ? (rootPos - _prevRootPosWorld) / dt : glm::vec3(0.f);

        int lf = _db.LeftFoot;
        int rf = _db.RightFoot;
        int hd = _db.Head;
        int lh = _db.LeftHand;
        int rh = _db.RightHand;

        auto safePos = [&](int idx) -> glm::vec3 {
            if (idx >= 0 && idx < int(poseNow.Pos.size())) return poseNow.Pos[idx];
            return rootPos;
        };

        glm::vec3 lfp = safePos(lf);
        glm::vec3 rfp = safePos(rf);
        glm::vec3 hdp = safePos(hd);
        glm::vec3 lhp = safePos(lh);
        glm::vec3 rhp = safePos(rh);

        glm::vec3 lfv = (_hasPrevKinematics && dt > 0.f) ? (lfp - _prevLeftFootPosWorld) / dt : glm::vec3(0.f);
        glm::vec3 rfv = (_hasPrevKinematics && dt > 0.f) ? (rfp - _prevRightFootPosWorld) / dt : glm::vec3(0.f);
        glm::vec3 hdv = (_hasPrevKinematics && dt > 0.f) ? (hdp - _prevHeadPosWorld) / dt : glm::vec3(0.f);
        glm::vec3 lhv = (_hasPrevKinematics && dt > 0.f) ? (lhp - _prevLeftHandPosWorld) / dt : glm::vec3(0.f);
        glm::vec3 rhv = (_hasPrevKinematics && dt > 0.f) ? (rhp - _prevRightHandPosWorld) / dt : glm::vec3(0.f);

        Feature q = BuildQueryFeature(
            rootPos,
            rootYaw,
            rootVel,
            _desiredFuturePosWorld,
            _desiredFutureFwdWorld,
            lfp,
            lfv,
            rfp,
            rfv,
            hdp,
            hdv,
            lhp,
            lhv,
            rhp,
            rhv);

        auto best  = _db.FindBestMatch(q, _weights, _currentIndex, _excludeWindow);
        _lastMatch = best;

        if (best.Index == _currentIndex) return;

        // compute alignment so that new db frame starts at current world root
        glm::vec3 rootPosDb = _db.RootPos[best.Index];
        glm::quat rootYawDb = _db.RootYaw[best.Index];

        Align newAlign;
        newAlign.Rot = rootYaw * glm::inverse(rootYawDb);
        newAlign.Pos = rootPos - newAlign.Rot * rootPosDb;

        float newTime = best.Index * _clip.FrameTime;

        if (_useInertialization) {
            auto poseA = sampleAligned(_dbTime, _align);
            auto poseB = sampleAligned(newTime, newAlign);

            int n = int(poseB.Pos.size());
            _inertialPosOffset.assign(n, glm::vec3(0.f));
            _inertialRotOffset.assign(n, glm::quat(1.f, 0.f, 0.f, 0.f));

            for (int i = 0; i < n; ++i) {
                _inertialPosOffset[i] = poseA.Pos[i] - poseB.Pos[i];
                _inertialRotOffset[i] = poseA.Rot[i] * glm::inverse(poseB.Rot[i]);
                _inertialRotOffset[i] = glm::normalize(_inertialRotOffset[i]);
            }

            _inertialActive = true;
            _blending       = false;

            _align  = newAlign;
            _dbTime = newTime;
        } else {
            _blending = (_blendDuration > 1e-4f);
            if (_blending) {
                _alignFrom = _align;
                _timeFrom  = _dbTime;
                _alignTo   = newAlign;
                _timeTo    = newTime;
                _blendT    = 0.f;
                _align     = newAlign;
                _dbTime    = _timeTo;
            } else {
                _align  = newAlign;
                _dbTime = newTime;
            }
        }

        _currentIndex = best.Index;
    }

    void CaseMotionMatching::computeWorldPose(float dt, GlobalPose & outPoseWorld) {
        if (! _clipLoaded) {
            outPoseWorld.Pos.clear();
            outPoseWorld.Rot.clear();
            return;
        }

        if (_blending) {
            _blendT += dt;
            float t = std::clamp(_blendT / std::max(_blendDuration, 1e-6f), 0.f, 1.f);
            _timeFrom += dt;
            _timeTo += dt;

            auto aPose = sampleAligned(_timeFrom, _alignFrom);
            auto bPose = sampleAligned(_timeTo, _alignTo);

            int n = int(aPose.Pos.size());
            outPoseWorld.Pos.resize(n);
            outPoseWorld.Rot.resize(n);
            for (int i = 0; i < n; ++i) {
                outPoseWorld.Pos[i] = glm::mix(aPose.Pos[i], bPose.Pos[i], t);
                outPoseWorld.Rot[i] = SafeSlerp(aPose.Rot[i], bPose.Rot[i], t);
            }

            if (t >= 1.f - 1e-5f) {
                _blending = false;
                _dbTime   = _timeTo;
            } else {
                _dbTime = _timeTo;
            }
            return;
        }

        // base pose
        outPoseWorld = sampleAligned(_dbTime, _align);

        // inertialization offsets
        if (_inertialActive && ! outPoseWorld.Pos.empty()) {
            float h     = std::max(_inertialHalfLife, 1e-3f);
            float alpha = 1.f - std::exp(-std::log(2.f) * dt / h); // exp decay step

            int n = int(outPoseWorld.Pos.size());
            if (int(_inertialPosOffset.size()) != n) {
                _inertialPosOffset.assign(n, glm::vec3(0.f));
                _inertialRotOffset.assign(n, glm::quat(1.f, 0.f, 0.f, 0.f));
            }

            bool done = true;
            for (int i = 0; i < n; ++i) {
                outPoseWorld.Pos[i] += _inertialPosOffset[i];
                outPoseWorld.Rot[i] = glm::normalize(_inertialRotOffset[i] * outPoseWorld.Rot[i]);

                _inertialPosOffset[i] = glm::mix(_inertialPosOffset[i], glm::vec3(0.f), alpha);
                _inertialRotOffset[i] = glm::normalize(glm::slerp(_inertialRotOffset[i], glm::quat(1.f, 0.f, 0.f, 0.f), alpha));

                if (glm::dot(_inertialPosOffset[i], _inertialPosOffset[i]) > 1e-6f) done = false;

            }
            if (done) _inertialActive = false;
        }
    }

    bool CaseMotionMatching::detectFootContact(
        glm::vec3 const & rootPosWorld,
        glm::quat const & rootYawWorld,
        glm::vec3 const & footPosWorld,
        glm::vec3 const & footVelWorld) const {
        glm::quat inv = glm::inverse(rootYawWorld);
        glm::vec3 pL  = inv * (footPosWorld - rootPosWorld);
        glm::vec3 vL  = inv * footVelWorld;

        glm::vec2 hv(vL.x, vL.z);
        bool      low  = std::abs(pL.y) < _footLockHeightThresh;
        bool      slow = glm::dot(hv, hv) < _footLockVelThresh * _footLockVelThresh;
        return low && slow;
    }

    void CaseMotionMatching::applyFootLock(
        float             dt,
        glm::vec3 const & rootPosWorld,
        glm::quat const & rootYawWorld,
        GlobalPose &      ioPoseWorld) {
        if (! _enableFootLock || ! _hasPrevKinematics || dt <= 0.f) return;
        if (ioPoseWorld.Pos.empty()) return;

        int lf = _db.LeftFoot;
        int rf = _db.RightFoot;
        if (lf < 0 || rf < 0) return;
        if (lf >= int(ioPoseWorld.Pos.size()) || rf >= int(ioPoseWorld.Pos.size())) return;

        glm::vec3 lfp = ioPoseWorld.Pos[lf];
        glm::vec3 rfp = ioPoseWorld.Pos[rf];
        glm::vec3 lfv = (lfp - _prevLeftFootPosWorld) / dt;
        glm::vec3 rfv = (rfp - _prevRightFootPosWorld) / dt;

        bool lc = detectFootContact(rootPosWorld, rootYawWorld, lfp, lfv);
        bool rc = detectFootContact(rootPosWorld, rootYawWorld, rfp, rfv);

        if (lc && ! _leftLocked) {
            _leftLocked       = true;
            _leftLockPosWorld = lfp;
        }
        if (! lc) _leftLocked = false;

        if (rc && ! _rightLocked) {
            _rightLocked       = true;
            _rightLockPosWorld = rfp;
        }
        if (! rc) _rightLocked = false;

        // If any locked, shift the whole pose by the lock error (XZ only).
        glm::vec3 delta(0.f);
        int       lockedCount = 0;
        if (_leftLocked) {
            delta += (_leftLockPosWorld - lfp);
            lockedCount++;
        }
        if (_rightLocked) {
            delta += (_rightLockPosWorld - rfp);
            lockedCount++;
        }

        if (lockedCount == 0) return;
        delta /= float(lockedCount);
        delta.y = 0.f;

        for (auto & p : ioPoseWorld.Pos) p += delta;

        // keep alignment coherent so next sampleAligned stays consistent
        _align.Pos += delta;
        _prevRootPosWorld += delta;
        _prevLeftFootPosWorld += delta;
        _prevRightFootPosWorld += delta;
        _prevHeadPosWorld += delta;
        _prevLeftHandPosWorld += delta;
        _prevRightHandPosWorld += delta;
    }

    Common::CaseRenderResult CaseMotionMatching::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        if (_reloadRequested) {
            _reloadRequested = false;
            loadBVH(std::filesystem::path(_bvhPathBuf.data()));
        }

        float dt = Engine::GetDeltaTime();
        if (_paused) dt = 0.f;

        if (_clipLoaded && dt > 0.f) {
            _dbTime += dt;
            _currentIndex = int(std::floor(_dbTime / _clip.FrameTime)) % std::max(1, _clip.FrameCount());
        }

        if (_clipLoaded && dt > 0.f) updateController(dt);
        if (_clipLoaded && dt > 0.f) updateMatching(dt);

        GlobalPose poseWorld;
        computeWorldPose(std::max(dt, 1e-6f), poseWorld);

        if (_clipLoaded && ! poseWorld.Pos.empty()) {
            glm::vec3 rootPos = poseWorld.Pos[_clip.Skeleton.Root];
            glm::quat rootYaw = ExtractYaw(poseWorld.Rot[_clip.Skeleton.Root]);

            applyFootLock(dt, rootPos, rootYaw, poseWorld);
        }

        // Update "prev" kinematics for query.
        if (_clipLoaded && ! poseWorld.Pos.empty()) {
            int root = _clip.Skeleton.Root;
            int lf   = _db.LeftFoot;
            int rf   = _db.RightFoot;
            int hd   = _db.Head;
            int lh   = _db.LeftHand;
            int rh   = _db.RightHand;

            auto safePos = [&](int idx) -> glm::vec3 {
                if (idx >= 0 && idx < int(poseWorld.Pos.size())) return poseWorld.Pos[idx];
                return poseWorld.Pos[root];
            };

            _prevRootPosWorld      = poseWorld.Pos[root];
            _prevLeftFootPosWorld  = safePos(lf);
            _prevRightFootPosWorld = safePos(rf);
            _prevHeadPosWorld      = safePos(hd);
            _prevLeftHandPosWorld  = safePos(lh);
            _prevRightHandPosWorld = safePos(rh);
            _hasPrevKinematics     = true;
        }

        // Build bone matrices
        _boneMatrices.fill(glm::mat4(1.f));
        if (_clipLoaded && ! poseWorld.Pos.empty()) {
            int n = std::min<int>(int(poseWorld.Pos.size()), int(kMaxBones));
            for (int i = 0; i < n; ++i) {
                glm::mat4 T      = glm::translate(glm::mat4(1.f), poseWorld.Pos[i]);
                glm::mat4 R      = glm::mat4_cast(poseWorld.Rot[i]);
                _boneMatrices[i] = T * R;
            }
        }

        // Skeleton line vertices
        if (_clipLoaded && _showSkeleton && ! poseWorld.Pos.empty()) {
            std::vector<glm::vec3> lines;
            lines.reserve(_clip.Skeleton.Joints.size() * 2);
            for (int j = 0; j < int(_clip.Skeleton.Joints.size()); ++j) {
                int p = _clip.Skeleton.Joints[j].Parent;
                if (p < 0) continue;
                lines.push_back(poseWorld.Pos[p]);
                lines.push_back(poseWorld.Pos[j]);
            }
            _skeletonLines.UpdateVertexBuffer(
                "position",
                Engine::make_span_bytes(std::span<glm::vec3 const>(lines.data(), lines.size())));
        }

        // Trajectory lines
        if (_clipLoaded && _showTrajectory && _hasPrevKinematics) {
            std::vector<glm::vec3> traj;
            traj.reserve(12);

            glm::vec3 root = _prevRootPosWorld;
            for (int i = 0; i < 3; ++i) {
                traj.push_back(root);
                traj.push_back(_desiredFuturePosWorld[i]);
            }
            for (int i = 0; i < 3; ++i) {
                glm::vec3 p   = _desiredFuturePosWorld[i];
                glm::vec3 fwd = _desiredFutureFwdWorld[i];
                fwd.y         = 0.f;
                if (glm::dot(fwd, fwd) < 1e-6f) fwd = { 0.f, 0.f, 1.f };
                fwd = glm::normalize(fwd);
                traj.push_back(p);
                traj.push_back(p + fwd * 0.25f);
            }

            _trajectoryLines.UpdateVertexBuffer(
                "position",
                Engine::make_span_bytes(std::span<glm::vec3 const>(traj.data(), traj.size())));
        }

        // Render
        _frame.Resize(desiredSize);

        bool      hasRoot      = _clipLoaded && _clip.Skeleton.Root >= 0 && _clip.Skeleton.Root < int(poseWorld.Pos.size());
        glm::vec3 rootPosWorld = hasRoot ? poseWorld.Pos[_clip.Skeleton.Root] : glm::vec3(0.f);

        _cameraManager.Update(_camera);

        if (_followRoot && hasRoot) {
            if (! _hasPrevFollowRootPosWorld) {
                _prevFollowRootPosWorld    = rootPosWorld;
                _hasPrevFollowRootPosWorld = true;
            } else {
                glm::vec3 delta = rootPosWorld - _prevFollowRootPosWorld;
                _camera.Eye += delta;
                _camera.Target += delta;
                _prevFollowRootPosWorld = rootPosWorld;
            }
        } else {
            _hasPrevFollowRootPosWorld = false;
        }

        glm::mat4 P = _camera.GetProjectionMatrix(float(desiredSize.first) / float(desiredSize.second));
        glm::mat4 V = _camera.GetViewMatrix();

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glClearColor(0.12f, 0.12f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (_clipLoaded && _showMesh) {
            _skinnedProgram.GetUniforms().SetByName("u_Projection", P);
            _skinnedProgram.GetUniforms().SetByName("u_View", V);
            _skinnedProgram.GetUniforms().SetByName("u_Color", glm::vec3(0.75f, 0.75f, 0.85f));

            _bonesBlock.u_Bones = _boneMatrices;
            _bonesUbo.Update(_bonesBlock);

            _mesh.Draw({ _skinnedProgram.Use() });
        }

        glDisable(GL_CULL_FACE);
        glLineWidth(2.f);

        if (_clipLoaded && _showSkeleton) {
            _flatProgram.GetUniforms().SetByName("u_Projection", P);
            _flatProgram.GetUniforms().SetByName("u_View", V);
            _flatProgram.GetUniforms().SetByName("u_Color", glm::vec3(0.1f, 0.95f, 0.2f));
            _skeletonLines.Draw({ _flatProgram.Use() });
        }

        if (_clipLoaded && _showTrajectory) {
            _flatProgram.GetUniforms().SetByName("u_Projection", P);
            _flatProgram.GetUniforms().SetByName("u_View", V);
            _flatProgram.GetUniforms().SetByName("u_Color", glm::vec3(0.95f, 0.6f, 0.2f));
            _trajectoryLines.Draw({ _flatProgram.Use() });
        }

        glLineWidth(1.f);
        glDisable(GL_DEPTH_TEST);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }
} // namespace VCX::Labs::Animation
