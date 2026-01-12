#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Engine/Camera.hpp"
#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Engine/GL/resource.hpp"
#include "Engine/GL/UniformBlock.hpp"

#include "Labs/Common/ICase.h"
#include "Labs/Common/OrbitCameraManager.h"

#include "Labs/4-Animation/BVHLoader.h"
#include "Labs/4-Animation/MotionMatching.h"

namespace VCX::Labs::Animation {

    static constexpr std::size_t   kMaxBones          = 256;
    static constexpr std::uint32_t kBonesBindingPoint = 1;

    struct alignas(16) BonesBlock {
        std::array<glm::mat4, kMaxBones> u_Bones {};
    };

    class CaseMotionMatching : public Common::ICase {
    public:
        CaseMotionMatching();

        struct SkinnedVertex {
            glm::vec3 Position;
            glm::vec3 Normal;
            glm::vec4 Bone;   // indices stored as floats
            glm::vec4 Weight; // weights
        };

        std::string_view const   GetName() override { return "Motion Matching"; }
        void                     OnSetupPropsUI() override;
        Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        struct Align {
            glm::quat Rot { 1.f, 0.f, 0.f, 0.f };
            glm::vec3 Pos { 0.f };
        };

        void loadBVH(std::filesystem::path const & path);
        void rebuildDatabase();
        void buildProceduralMesh();

        void updateController(float dt);
        void updateMatching(float dt);

        GlobalPose sampleAligned(float dbTime, Align const & align) const;
        void       computeWorldPose(float dt, GlobalPose & outPoseWorld);

        void fitCameraToPose(GlobalPose const & poseWorld);

        // foot contact + lock
        bool detectFootContact(
            glm::vec3 const & rootPosWorld,
            glm::quat const & rootYawWorld,
            glm::vec3 const & footPosWorld,
            glm::vec3 const & footVelWorld) const;

        void applyFootLock(
            float             dt,
            glm::vec3 const & rootPosWorld,
            glm::quat const & rootYawWorld,
            GlobalPose &      ioPoseWorld);

        void resetFollowRootState() {
            _hasPrevFollowRootPosWorld = false;
            _prevFollowRootPosWorld    = glm::vec3(0.f);
        }

    private:
        // Assets / data
        BVHClip        _clip;
        bool           _clipLoaded { false };
        MotionDatabase _db;

        // Runtime state
        Align _align;
        float _dbTime { 0.f };
        int   _currentIndex { 0 };

        // Inertialization (better transition)
        bool                   _useInertialization { true };
        float                  _inertialHalfLife { 0.12f };
        bool                   _inertialActive { false };
        std::vector<glm::vec3> _inertialPosOffset; // world-space offsets
        std::vector<glm::quat> _inertialRotOffset; // rotOffset * newRot

        // (fallback) simple blend
        bool  _blending { false };
        Align _alignFrom;
        Align _alignTo;
        float _timeFrom { 0.f };
        float _timeTo { 0.f };
        float _blendT { 0.f };
        float _blendDuration { 0.18f };

        // Search settings
        float _searchTimer { 0.f };
        float _searchInterval { 0.10f };
        int   _excludeWindow { 10 };

        Feature     _weights {};
        MatchResult _lastMatch {};

        // Controller (keyboard)
        float _walkSpeed { 1.2f };
        float _runSpeed { 2.4f };
        float _turnSpeedDeg { 140.f };
        float _desiredYaw { 0.f };

        glm::vec3                _desiredVelWorld { 0.f };
        std::array<glm::vec3, 3> _desiredFuturePosWorld {
            glm::vec3(0.f), glm::vec3(0.f), glm::vec3(0.f)
        };
        std::array<glm::vec3, 3> _desiredFutureFwdWorld {
            glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 1.f)
        };

        // Kinematics cache for query
        glm::vec3 _prevRootPosWorld { 0.f };
        glm::vec3 _prevLeftFootPosWorld { 0.f };
        glm::vec3 _prevRightFootPosWorld { 0.f };
        glm::vec3 _prevHeadPosWorld { 0.f };
        glm::vec3 _prevLeftHandPosWorld { 0.f };
        glm::vec3 _prevRightHandPosWorld { 0.f };
        bool      _hasPrevKinematics { false };

        // Foot lock state
        bool      _enableFootLock { true };
        float     _footLockVelThresh { 0.18f };
        float     _footLockHeightThresh { 0.08f };
        bool      _leftLocked { false };
        bool      _rightLocked { false };
        glm::vec3 _leftLockPosWorld { 0.f };
        glm::vec3 _rightLockPosWorld { 0.f };

        // UI toggles
        bool _paused { false };
        bool _showMesh { true };
        bool _showSkeleton { true };
        bool _showTrajectory { true };
        bool _followRoot { true };

        // Rendering
        Engine::GL::UniqueProgram     _flatProgram;
        Engine::GL::UniqueProgram     _skinnedProgram;
        Engine::GL::UniqueRenderFrame _frame;

        Engine::Camera _camera {
            .Eye    = glm::vec3(-3.f, 2.2f, 3.f),
            .Target = glm::vec3(0.f, 1.f, 0.f),
        };
        Common::OrbitCameraManager _cameraManager;

        Engine::GL::UniqueRenderItem        _skeletonLines;
        Engine::GL::UniqueRenderItem        _trajectoryLines;
        Engine::GL::UniqueIndexedRenderItem _mesh;

        std::vector<SkinnedVertex> _meshVertices;
        std::vector<std::uint32_t> _meshIndices;

        std::array<glm::mat4, kMaxBones>           _boneMatrices {};
        BonesBlock                                 _bonesBlock {};
        Engine::GL::UniqueUniformBlock<BonesBlock> _bonesUbo;

        // UI state
        std::array<char, 256> _bvhPathBuf {};
        bool                  _reloadRequested { false };

        // Follow-root camera state
        bool      _hasPrevFollowRootPosWorld { false };
        glm::vec3 _prevFollowRootPosWorld { 0.f };
    };

} // namespace VCX::Labs::Animation
