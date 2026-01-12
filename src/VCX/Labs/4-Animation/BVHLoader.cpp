#include "Labs/4-Animation/BVHLoader.h"

#include <cmath>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <glm/gtc/constants.hpp>

namespace VCX::Labs::Animation {
    namespace {
        std::vector<std::string> Tokenize(std::string const & text) {
            std::vector<std::string> tokens;
            std::string cur;
            auto flush = [&]() {
                if (!cur.empty()) {
                    tokens.push_back(cur);
                    cur.clear();
                }
            };
            for (char c : text) {
                if (c == '{' || c == '}' ) {
                    flush();
                    tokens.emplace_back(1, c);
                } else if (std::isspace(static_cast<unsigned char>(c))) {
                    flush();
                } else {
                    cur.push_back(c);
                }
            }
            flush();
            return tokens;
        }

        std::string StripColon(std::string s) {
            if (!s.empty() && s.back() == ':') s.pop_back();
            return s;
        }

        float ParseFloat(std::string const & s) {
            size_t idx = 0;
            float v = std::stof(s, &idx);
            if (idx != s.size()) throw std::runtime_error("BVH parse: invalid float: " + s);
            return v;
        }

        int ParseInt(std::string const & s) {
            size_t idx = 0;
            int v = std::stoi(s, &idx);
            if (idx != s.size()) throw std::runtime_error("BVH parse: invalid int: " + s);
            return v;
        }

        BVHChannel ParseChannel(std::string const & s) {
            static const std::unordered_map<std::string, BVHChannel> map = {
                {"Xposition", BVHChannel::Xposition},
                {"Yposition", BVHChannel::Yposition},
                {"Zposition", BVHChannel::Zposition},
                {"Xrotation", BVHChannel::Xrotation},
                {"Yrotation", BVHChannel::Yrotation},
                {"Zrotation", BVHChannel::Zrotation},
            };
            auto it = map.find(s);
            if (it == map.end()) throw std::runtime_error("BVH parse: unknown channel: " + s);
            return it->second;
        }

        glm::quat EulerToQuat(std::vector<BVHChannel> const & channels, std::vector<float> const & frame, int start) {
            // Apply rotations in the order given by BVH channels, around local axes.
            glm::quat q(1.f, 0.f, 0.f, 0.f);
            for (std::size_t i = 0; i < channels.size(); ++i) {
                auto ch = channels[i];
                if (ch == BVHChannel::Xrotation || ch == BVHChannel::Yrotation || ch == BVHChannel::Zrotation) {
                    float deg = frame[start + int(i)];
                    float rad = glm::radians(deg);
                    glm::vec3 axis(0.f);
                    if (ch == BVHChannel::Xrotation) axis = {1.f, 0.f, 0.f};
                    if (ch == BVHChannel::Yrotation) axis = {0.f, 1.f, 0.f};
                    if (ch == BVHChannel::Zrotation) axis = {0.f, 0.f, 1.f};
                    q = q * glm::angleAxis(rad, axis);
                }
            }
            return q;
        }
    }

    glm::quat ExtractYaw(glm::quat const & q) {
        glm::vec3 fwd = q * glm::vec3(0.f, 0.f, 1.f);
        fwd.y = 0.f;
        // glm 1.0.0 used by xmake does not provide glm::length2; use dot(fwd,fwd) instead.
        if (glm::dot(fwd, fwd) < 1e-8f) return glm::quat(1.f, 0.f, 0.f, 0.f);
        fwd = glm::normalize(fwd);
        float yaw = std::atan2(fwd.x, fwd.z);
        return glm::angleAxis(yaw, glm::vec3(0.f, 1.f, 0.f));
    }

    BVHClip LoadBVH(std::filesystem::path const & path) {
        // Read file.
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Failed to open BVH: " + path.string());
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto tokens = Tokenize(content);
        std::size_t p = 0;
        auto expect = [&](std::string const & t) {
            if (p >= tokens.size() || tokens[p] != t)
                throw std::runtime_error("BVH parse: expected '" + t + "' near token " + (p < tokens.size() ? tokens[p] : std::string("<eof>")));
            ++p;
        };
        auto next = [&]() -> std::string {
            if (p >= tokens.size()) throw std::runtime_error("BVH parse: unexpected EOF");
            return tokens[p++];
        };

        BVHClip clip;
        int channelCursor = 0;

        auto parseNode = [&](auto && self, int parentIdx, std::string const & kind) -> int {
            // kind is ROOT/JOINT/End
            std::string name;
            if (kind == "End") {
                // Expect "Site".
                expect("Site");
                name = "EndSite";
            } else {
                name = next();
            }
            int myIdx = int(clip.Skeleton.Joints.size());
            clip.Skeleton.Joints.push_back(BVHJoint{ .Name = name, .Parent = parentIdx });
            if (parentIdx >= 0) clip.Skeleton.Joints[parentIdx].Children.push_back(myIdx);

            expect("{");
            while (true) {
                std::string t = next();
                if (t == "}") break;
                if (t == "OFFSET") {
                    float x = ParseFloat(next());
                    float y = ParseFloat(next());
                    float z = ParseFloat(next());
                    clip.Skeleton.Joints[myIdx].Offset = {x, y, z};
                } else if (t == "CHANNELS") {
                    int n = ParseInt(next());
                    clip.Skeleton.Joints[myIdx].Channels.clear();
                    clip.Skeleton.Joints[myIdx].Channels.reserve(n);
                    clip.Skeleton.Joints[myIdx].ChannelStart = channelCursor;
                    for (int i = 0; i < n; ++i) {
                        clip.Skeleton.Joints[myIdx].Channels.push_back(ParseChannel(next()));
                    }
                    channelCursor += n;
                } else if (t == "JOINT") {
                    self(self, myIdx, "JOINT");
                } else if (t == "End") {
                    self(self, myIdx, "End");
                } else {
                    throw std::runtime_error("BVH parse: unexpected token in node: " + t);
                }
            }
            return myIdx;
        };

        // Header
        expect("HIERARCHY");
        expect("ROOT");
        clip.Skeleton.Root = parseNode(parseNode, -1, "ROOT");
        clip.TotalChannels = channelCursor;

        // Motion
        expect("MOTION");
        {
            auto t = StripColon(next());
            if (t != "Frames") throw std::runtime_error("BVH parse: expected Frames");
            std::string maybeColon = tokens[p];
            if (maybeColon == ":") ++p; // if colon was tokenized (it usually isn't)
            int frames = ParseInt(next());

            auto t2 = StripColon(next());
            if (t2 != "Frame") throw std::runtime_error("BVH parse: expected Frame Time");
            auto t3 = StripColon(next());
            if (t3 != "Time") throw std::runtime_error("BVH parse: expected Frame Time");
            float frameTime = ParseFloat(next());

            clip.FrameTime = frameTime;
            clip.Frames.resize(frames);
            for (int f = 0; f < frames; ++f) {
                clip.Frames[f].resize(clip.TotalChannels);
                for (int c = 0; c < clip.TotalChannels; ++c) {
                    clip.Frames[f][c] = ParseFloat(next());
                }
            }
        }

        return clip;
    }
}
