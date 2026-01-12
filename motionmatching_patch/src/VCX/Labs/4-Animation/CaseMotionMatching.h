#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "Engine/Camera.hpp"
#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/OrbitCameraManager.h"
#include "Labs/4-Animation/BVHLoader.h"
#include "Labs/4-Animation/MotionMatching.h"

namespace VCX::Labs::Animation {

    class CaseMotionMatching : public Common::ICase {
    public:
        CaseMotionMatching();

        std::string_view const GetName() override { return "Motion Matching"; }
        void                     OnSetupPropsUI() override;
        Common::CaseRenderResult  OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        void                     OnProcessInput(ImVec2 const & pos) override;

        struct SkinnedVertex {
            glm::vec3 Position;
            glm::vec3 Normal;
            glm::vec4 Bone;
            glm::vec4 Weight;
        };

    private:
        struct Align {
            glm::quat Rot { 1.f, 0.f, 0.f, 0.f };
            glm::vec3 Pos { 0.f };
        };

        static constexpr std::size_t kMaxBones = 256;

        void loadBVH(std::filesystem::path const & path);
        void rebuildDatabase();
        void buildProceduralMesh();

        void updateController(float dt);
        void updateMatching(float dt);

        GlobalPose sampleAligned(float dbTime, Align const & align) const;
        void       computeWorldPose(float dt, GlobalPose & outPoseWorld);

        // Assets / data
        BVHClip         _clip;
        bool            _clipLoaded { false };
        MotionDatabase  _db;

        // Runtime state
        Align _align;
        float _dbTime { 0.f };
        int   _currentIndex { 0 };

        bool  _blending { false };
        Align _alignFrom;
        Align _alignTo;
        float _timeFrom { 0.f };
        float _timeTo   { 0.f };
        float _blendT   { 0.f };
        float _blendDuration { 0.18f };

        float _searchTimer { 0.f };
        float _searchInterval { 0.10f };
        int   _excludeWindow { 10 };

        Feature _weights;
        MatchResult _lastMatch;

        // Controller (keyboard)
        float _walkSpeed { 1.2f };
        float _runSpeed  { 2.4f };
        float _turnSpeedDeg { 140.f };
        float _desiredYaw { 0.f };

        glm::vec3 _desiredVelWorld { 0.f };
        std::array<glm::vec3, 3> _desiredFuturePosWorld { glm::vec3(0.f), glm::vec3(0.f), glm::vec3(0.f) };
        std::array<glm::vec3, 3> _desiredFutureFwdWorld { glm::vec3(0.f,0.f,1.f), glm::vec3(0.f,0.f,1.f), glm::vec3(0.f,0.f,1.f) };

        glm::vec3 _prevRootPosWorld { 0.f };
        glm::vec3 _prevLeftFootPosWorld { 0.f };
        glm::vec3 _prevRightFootPosWorld { 0.f };
        bool      _hasPrevKinematics { false };

        bool _paused { false };
        bool _showMesh { true };
        bool _showSkeleton { true };
        bool _showTrajectory { true };

        // Rendering
        Engine::GL::UniqueProgram     _flatProgram;
        Engine::GL::UniqueProgram     _skinnedProgram;
        Engine::GL::UniqueRenderFrame _frame;
        Engine::Camera                _camera { .Eye = glm::vec3(-3.f, 2.2f, 3.f), .Target = glm::vec3(0.f, 1.f, 0.f) };
        Common::OrbitCameraManager    _cameraManager;

        Engine::GL::UniqueRenderItem        _skeletonLines;
        Engine::GL::UniqueRenderItem        _trajectoryLines;
        Engine::GL::UniqueIndexedRenderItem _mesh;

        std::vector<SkinnedVertex>          _meshVertices;
        std::vector<std::uint32_t>          _meshIndices;

        std::array<glm::mat4, kMaxBones>    _boneMatrices;

        // UI state
        std::array<char, 256> _bvhPathBuf {};
        bool        _reloadRequested { false };
    };

} // namespace VCX::Labs::Animation
