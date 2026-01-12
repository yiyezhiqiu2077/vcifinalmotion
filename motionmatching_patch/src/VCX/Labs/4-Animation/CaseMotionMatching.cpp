#include "Labs/4-Animation/CaseMotionMatching.h"

#include <algorithm>
#include <cstdio>
#include <cmath>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
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
            return glm::slerp(a, b, t);
        }

        void AppendBox(
            std::vector<CaseMotionMatching::SkinnedVertex> & verts,
            std::vector<std::uint32_t> & indices,
            int boneIndex,
            glm::mat3 const & R,
            float length,
            float r) {
            // Box in canonical space: x/y in [-r,r], z in [0,length]
            struct Face {
                glm::vec3 n;
                glm::vec3 v0, v1, v2, v3;
            };
            float z0 = 0.f;
            float z1 = length;
            // 6 faces
            std::array<Face, 6> faces = {
                Face{ { 1, 0, 0 }, { r,-r,z0 },{ r, r,z0 },{ r, r,z1 },{ r,-r,z1 } },
                Face{ {-1, 0, 0 }, {-r,-r,z1 },{-r, r,z1 },{-r, r,z0 },{-r,-r,z0 } },
                Face{ { 0, 1, 0 }, {-r, r,z0 },{ r, r,z0 },{ r, r,z1 },{-r, r,z1 } },
                Face{ { 0,-1, 0 }, {-r,-r,z1 },{ r,-r,z1 },{ r,-r,z0 },{-r,-r,z0 } },
                Face{ { 0, 0, 1 }, {-r,-r,z1 },{-r, r,z1 },{ r, r,z1 },{ r,-r,z1 } },
                Face{ { 0, 0,-1 }, {-r,-r,z0 },{ r,-r,z0 },{ r, r,z0 },{-r, r,z0 } },
            };

            glm::vec4 bone(float(boneIndex), 0.f, 0.f, 0.f);
            glm::vec4 w(1.f, 0.f, 0.f, 0.f);

            for (auto const & f : faces) {
                std::uint32_t base = std::uint32_t(verts.size());
                glm::vec3 n = glm::normalize(R * f.n);
                auto addV = [&](glm::vec3 const & p) {
                    verts.push_back({
                        .Position = R * p,
                        .Normal   = n,
                        .Bone     = bone,
                        .Weight   = w,
                    });
                };
                addV(f.v0); addV(f.v1); addV(f.v2); addV(f.v3);
                indices.push_back(base + 0);
                indices.push_back(base + 1);
                indices.push_back(base + 2);
                indices.push_back(base + 0);
                indices.push_back(base + 2);
                indices.push_back(base + 3);
            }
        }

        glm::mat3 RotationFromTo(glm::vec3 from, glm::vec3 to) {
            from = glm::normalize(from);
            to = glm::normalize(to);
            if (glm::length2(from) < 1e-8f || glm::length2(to) < 1e-8f) return glm::mat3(1.f);
            glm::quat q = glm::rotation(from, to);
            return glm::mat3_cast(q);
        }
    }

    CaseMotionMatching::CaseMotionMatching() :
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
        _mesh(
            Engine::GL::VertexLayout()
                .Add<SkinnedVertex>("vtx", Engine::GL::DrawFrequency::Static)
                .At(0, &SkinnedVertex::Position)
                .At(1, &SkinnedVertex::Normal)
                .At(2, &SkinnedVertex::Bone)
                .At(3, &SkinnedVertex::Weight),
            Engine::GL::PrimitiveType::Triangles) {

        // Default weights (common choice: emphasize trajectory more than feet)
        _weights.fill(1.f);
        // Root vel
        _weights[0] = 2.f; _weights[1] = 2.f;
        // Trajectory positions
        for (int i = 2; i < 8; ++i) _weights[i] = 3.f;
        // Facing
        for (int i = 8; i < 14; ++i) _weights[i] = 1.5f;
        // Feet
        for (int i = 14; i < 26; ++i) _weights[i] = 1.f;

        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_camera);

        std::snprintf(_bvhPathBuf.data(), _bvhPathBuf.size(), "%s", "assets/mocap/mm_synthetic_walk_turn.bvh");
        loadBVH(std::filesystem::path(_bvhPathBuf.data()));
    }

    void CaseMotionMatching::loadBVH(std::filesystem::path const & path) {
        try {
            _clip = LoadBVH(path);
            _clipLoaded = true;
            rebuildDatabase();
            buildProceduralMesh();
            _align = Align{};
            _dbTime = 0.f;
            _currentIndex = 0;
            _lastMatch = {0, 0.f};
            _hasPrevKinematics = false;
        } catch (std::exception const & e) {
            _clipLoaded = false;
            (void)e;
        }
    }

    void CaseMotionMatching::rebuildDatabase() {
        if (!_clipLoaded) return;
        _db.Build(_clip);
    }

    void CaseMotionMatching::buildProceduralMesh() {
        _meshVertices.clear();
        _meshIndices.clear();
        if (!_clipLoaded) return;

        // Build a rigid "skinned" mesh: one box per bone segment (joint -> first child)
        float const radius = 0.03f;
        for (int j = 0; j < int(_clip.Skeleton.Joints.size()) && j < int(kMaxBones); ++j) {
            auto const & joint = _clip.Skeleton.Joints[j];
            if (joint.Children.empty()) continue;
            int child = joint.Children.front();
            glm::vec3 off = _clip.Skeleton.Joints[child].Offset;
            float len = glm::length(off);
            if (len < 1e-4f) continue;
            glm::vec3 dir = off / len;
            glm::mat3 R = RotationFromTo(glm::vec3(0.f, 0.f, 1.f), dir);
            AppendBox(_meshVertices, _meshIndices, j, R, len, radius);
        }

        _mesh.UpdateVertexBuffer("vtx", Engine::make_span_bytes<SkinnedVertex>(_meshVertices));
        _mesh.UpdateElementBuffer(_meshIndices);
    }

    void CaseMotionMatching::OnSetupPropsUI() {
        ImGui::TextWrapped("W/S: forward/backward   A/D: turn   Shift: run\nMotion matching chooses the next frame from the database by matching trajectory + feet + velocity.");
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("BVH", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputText("BVH Path", _bvhPathBuf.data(), _bvhPathBuf.size());
            if (ImGui::Button("Reload BVH")) _reloadRequested = true;
            ImGui::SameLine();
            ImGui::TextDisabled(_clipLoaded ? "(loaded)" : "(load failed)");
            ImGui::Text("Frames: %d  dt: %.4f", _clipLoaded ? _clip.FrameCount() : 0, _clipLoaded ? _clip.FrameTime : 0.f);
        }
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Runtime", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button(_paused ? "Resume" : "Pause")) _paused = !_paused;
            ImGui::SameLine();
            ImGui::Checkbox("Mesh", &_showMesh);
            ImGui::SameLine();
            ImGui::Checkbox("Skeleton", &_showSkeleton);
            ImGui::SameLine();
            ImGui::Checkbox("Trajectory", &_showTrajectory);

            ImGui::SliderFloat("Walk Speed", &_walkSpeed, 0.2f, 3.0f);
            ImGui::SliderFloat("Run Speed", &_runSpeed, 0.5f, 6.0f);
            ImGui::SliderFloat("Turn Speed (deg/s)", &_turnSpeedDeg, 30.f, 300.f);

            ImGui::SliderFloat("Match Interval (s)", &_searchInterval, 0.02f, 0.30f);
            ImGui::SliderInt("Exclude Window (frames)", &_excludeWindow, 0, 60);
            ImGui::SliderFloat("Blend Duration (s)", &_blendDuration, 0.0f, 0.5f);

            ImGui::Text("Current frame: %d", _currentIndex);
            ImGui::Text("Last match: #%d  cost: %.4f", _lastMatch.Index, _lastMatch.Cost);
        }
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Feature Weights")) {
            ImGui::TextDisabled("(26D) RootVel(2), TrajPos(6), TrajFacing(6), LFoot(pos+vel 6), RFoot(pos+vel 6)");
            ImGui::SliderFloat2("RootVel", &_weights[0], 0.f, 10.f);
            ImGui::SliderFloat("TrajPos", &_weights[2], 0.f, 10.f);
            for (int i = 3; i < 8; ++i) _weights[i] = _weights[2];
            ImGui::SliderFloat("TrajFacing", &_weights[8], 0.f, 10.f);
            for (int i = 9; i < 14; ++i) _weights[i] = _weights[8];
            ImGui::SliderFloat("Feet", &_weights[14], 0.f, 10.f);
            for (int i = 15; i < 26; ++i) _weights[i] = _weights[14];
        }
    }

    void CaseMotionMatching::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_camera, pos);
    }

    GlobalPose CaseMotionMatching::sampleAligned(float dbTime, Align const & align) const {
        auto lp = SampleLocalPose(_clip, dbTime);
        auto gp = ComputeGlobalPose(_clip, lp);

        GlobalPose out;
        int n = int(gp.Pos.size());
        out.Pos.resize(n);
        out.Rot.resize(n);
        for (int i = 0; i < n; ++i) {
            out.Pos[i] = align.Pos + align.Rot * gp.Pos[i];
            out.Rot[i] = align.Rot * gp.Rot[i];
        }
        return out;
    }

    void CaseMotionMatching::updateController(float dt) {
        auto & io = ImGui::GetIO();
        bool W = ImGui::IsKeyDown(ImGuiKey_W);
        bool S = ImGui::IsKeyDown(ImGuiKey_S);
        bool A = ImGui::IsKeyDown(ImGuiKey_A);
        bool D = ImGui::IsKeyDown(ImGuiKey_D);
        bool shift = io.KeyShift;

        float yawRate = glm::radians(_turnSpeedDeg);
        if (A) _desiredYaw += yawRate * dt;
        if (D) _desiredYaw -= yawRate * dt;
        _desiredYaw = WrapAngle(_desiredYaw);

        float speed = 0.f;
        if (W) speed += (shift ? _runSpeed : _walkSpeed);
        if (S) speed -= 0.6f * (shift ? _runSpeed : _walkSpeed);
        glm::vec3 fwd = YawFromAngle(_desiredYaw) * glm::vec3(0.f, 0.f, 1.f);
        _desiredVelWorld = fwd * speed;

        // Desired future trajectory in world space (relative to current root).
        std::array<float, 3> const ts = { 0.2f, 0.4f, 0.6f };
        for (int i = 0; i < 3; ++i) {
            _desiredFuturePosWorld[i] = _prevRootPosWorld + _desiredVelWorld * ts[i];
            _desiredFutureFwdWorld[i] = fwd;
        }
    }

    void CaseMotionMatching::updateMatching(float dt) {
        if (!_clipLoaded || _db.Features.empty()) return;

        _searchTimer += dt;
        if (_searchTimer < _searchInterval) return;
        _searchTimer = 0.f;

        // Current pose in world (for building the query)
        auto poseNow = sampleAligned(_dbTime, _align);
        glm::vec3 rootPos = poseNow.Pos[_clip.Skeleton.Root];
        glm::quat rootYaw = ExtractYaw(poseNow.Rot[_clip.Skeleton.Root]);
        glm::vec3 rootVel = (_hasPrevKinematics && dt > 0.f) ? (rootPos - _prevRootPosWorld) / dt : glm::vec3(0.f);

        int lf = _db.LeftFoot;
        int rf = _db.RightFoot;
        glm::vec3 lfp = (lf >= 0 && lf < int(poseNow.Pos.size())) ? poseNow.Pos[lf] : rootPos;
        glm::vec3 rfp = (rf >= 0 && rf < int(poseNow.Pos.size())) ? poseNow.Pos[rf] : rootPos;
        glm::vec3 lfv = (_hasPrevKinematics && dt > 0.f) ? (lfp - _prevLeftFootPosWorld) / dt : glm::vec3(0.f);
        glm::vec3 rfv = (_hasPrevKinematics && dt > 0.f) ? (rfp - _prevRightFootPosWorld) / dt : glm::vec3(0.f);

        Feature q = BuildQueryFeature(
            rootPos,
            rootYaw,
            rootVel,
            _desiredFuturePosWorld,
            _desiredFutureFwdWorld,
            lfp,
            lfv,
            rfp,
            rfv);

        // Match in database space by converting rootVelWorld etc via BuildQueryFeature.
        auto best = _db.FindBestMatch(q, _weights, _currentIndex, _excludeWindow);
        _lastMatch = best;

        // Switch if improvement is meaningful, or if far away.
        if (best.Index != _currentIndex) {
            // Compute alignment so that new db frame starts at current world root
            glm::vec3 rootPosWorld = rootPos;
            glm::quat rootYawWorld = rootYaw;

            glm::vec3 rootPosDb = _db.RootPos[best.Index];
            glm::quat rootYawDb = _db.RootYaw[best.Index];

            Align newAlign;
            newAlign.Rot = rootYawWorld * glm::inverse(rootYawDb);
            newAlign.Pos = rootPosWorld - newAlign.Rot * rootPosDb;

            // Start blending
            _blending = (_blendDuration > 1e-4f);
            if (_blending) {
                _alignFrom = _align;
                _timeFrom  = _dbTime;
                _alignTo   = newAlign;
                _timeTo    = best.Index * _clip.FrameTime;
                _blendT    = 0.f;
                _align     = newAlign; // after blending finished
                _dbTime    = _timeTo;
            } else {
                _align = newAlign;
                _dbTime = best.Index * _clip.FrameTime;
            }

            _currentIndex = best.Index;
        }
    }

    void CaseMotionMatching::computeWorldPose(float dt, GlobalPose & outPoseWorld) {
        if (!_clipLoaded) {
            outPoseWorld.Pos.clear();
            outPoseWorld.Rot.clear();
            return;
        }

        if (_blending) {
            _blendT += dt;
            float t = std::clamp(_blendT / std::max(_blendDuration, 1e-6f), 0.f, 1.f);
            _timeFrom += dt;
            _timeTo   += dt;

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
                _dbTime = _timeTo;
            } else {
                _dbTime = _timeTo;
            }
        } else {
            outPoseWorld = sampleAligned(_dbTime, _align);
        }
    }

    Common::CaseRenderResult CaseMotionMatching::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        if (_reloadRequested) {
            _reloadRequested = false;
            loadBVH(std::filesystem::path(_bvhPathBuf.data()));
        }

        float dt = Engine::GetDeltaTime();
        if (_paused) dt = 0.f;

        if (_clipLoaded && !_paused) {
            // Advance playback
            _dbTime += dt;
            // Keep a reasonable index estimate for exclude-window.
            _currentIndex = int(std::floor(_dbTime / _clip.FrameTime)) % std::max(1, _clip.FrameCount());
        }

        if (_clipLoaded) {
            updateController(std::max(dt, 1e-6f));
            if (!_paused) updateMatching(std::max(dt, 1e-6f));
        }

        GlobalPose poseWorld;
        computeWorldPose(std::max(dt, 1e-6f), poseWorld);

        // Update "prev" kinematics for query.
        if (_clipLoaded && !poseWorld.Pos.empty()) {
            int root = _clip.Skeleton.Root;
            int lf = _db.LeftFoot;
            int rf = _db.RightFoot;
            glm::vec3 rootPos = poseWorld.Pos[root];
            glm::vec3 lfp = (lf >= 0 && lf < int(poseWorld.Pos.size())) ? poseWorld.Pos[lf] : rootPos;
            glm::vec3 rfp = (rf >= 0 && rf < int(poseWorld.Pos.size())) ? poseWorld.Pos[rf] : rootPos;
            _prevRootPosWorld = rootPos;
            _prevLeftFootPosWorld = lfp;
            _prevRightFootPosWorld = rfp;
            _hasPrevKinematics = true;
        }

        // Build bone matrices
        _boneMatrices.fill(glm::mat4(1.f));
        if (_clipLoaded && !poseWorld.Pos.empty()) {
            int n = std::min<int>(int(poseWorld.Pos.size()), int(kMaxBones));
            for (int i = 0; i < n; ++i) {
                glm::mat4 T = glm::translate(glm::mat4(1.f), poseWorld.Pos[i]);
                glm::mat4 R = glm::mat4_cast(poseWorld.Rot[i]);
                _boneMatrices[i] = T * R;
            }
        }

        // Skeleton line vertices
        if (_clipLoaded && _showSkeleton && !poseWorld.Pos.empty()) {
            std::vector<glm::vec3> lines;
            lines.reserve(_clip.Skeleton.Joints.size() * 2);
            for (int j = 0; j < int(_clip.Skeleton.Joints.size()); ++j) {
                int p = _clip.Skeleton.Joints[j].Parent;
                if (p < 0) continue;
                lines.push_back(poseWorld.Pos[p]);
                lines.push_back(poseWorld.Pos[j]);
            }
            _skeletonLines.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(lines));
        }

        // Trajectory lines
        if (_clipLoaded && _showTrajectory && _hasPrevKinematics) {
            std::vector<glm::vec3> traj;
            traj.reserve(8);
            glm::vec3 root = _prevRootPosWorld;
            for (int i = 0; i < 3; ++i) {
                traj.push_back(root);
                traj.push_back(_desiredFuturePosWorld[i]);
            }
            // facing indicators at each future point
            for (int i = 0; i < 3; ++i) {
                glm::vec3 p = _desiredFuturePosWorld[i];
                glm::vec3 fwd = _desiredFutureFwdWorld[i];
                fwd.y = 0.f;
                if (glm::length2(fwd) < 1e-6f) fwd = {0.f,0.f,1.f};
                fwd = glm::normalize(fwd);
                traj.push_back(p);
                traj.push_back(p + fwd * 0.25f);
            }
            _trajectoryLines.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(traj));
        }

        // Render
        _frame.Resize(desiredSize);
        _cameraManager.Update(_camera);

        glm::mat4 P = _camera.GetProjectionMatrix(float(desiredSize.first) / float(desiredSize.second));
        glm::mat4 V = _camera.GetViewMatrix();

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glClearColor(0.12f, 0.12f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw mesh
        if (_clipLoaded && _showMesh) {
            _skinnedProgram.GetUniforms().SetByName("u_Projection", P);
            _skinnedProgram.GetUniforms().SetByName("u_View", V);
            _skinnedProgram.GetUniforms().SetByName("u_Color", glm::vec3(0.75f, 0.75f, 0.85f));
            _skinnedProgram.GetUniforms().SetByName("u_Bones", _boneMatrices);
            _mesh.Draw({ _skinnedProgram.Use() });
        }

        // Draw skeleton + trajectory on top
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
            .Fixed = false,
            .Flipped = true,
            .Image = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }
}
