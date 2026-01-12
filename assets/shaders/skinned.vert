#version 410 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec4 a_Bone;   // indices (stored as floats)
layout(location = 3) in vec4 a_Weight; // weights

layout(location = 0) out vec3 v_Normal;

uniform mat4 u_Projection;
uniform mat4 u_View;

const int MAX_BONES = 256;

// 关键修改：GLSL 410 不支持 binding = N，所以删掉 binding，靠 C++ BindUniformBlock 来绑定
layout(std140) uniform Bones
{
    mat4 u_Bones[MAX_BONES];
};

void main() {
    ivec4 bi = ivec4(a_Bone);
    mat4 skin =
        a_Weight.x * u_Bones[bi.x] +
        a_Weight.y * u_Bones[bi.y] +
        a_Weight.z * u_Bones[bi.z] +
        a_Weight.w * u_Bones[bi.w];

    vec4 wp = skin * vec4(a_Position, 1.0);
    v_Normal = normalize(mat3(skin) * a_Normal);
    gl_Position = u_Projection * u_View * wp;
}
