#pragma once

#include "noise.hpp"
#include "sdf_base.hpp"
#include "sdf_test_rig.hpp"
#include <cmath>

// Placeholder terrain, written in metres (see SDFMetreBridge).
class SDFDummy : public SDFMetreBridge {
public:
    // Adds the centimetre-scale test shapes of sdf_test_rig.hpp (set before meshing starts).
    bool test_rig = false;

    // Debug: a square column in front of the main_scene spawn whose flat top lies exactly on a lattice
    // row (multiple of 64 LOD0 voxels), so the SDF is exactly 0.0 there at LODs 0-6.
    // 1 = exact top, 2 = top 1 cm higher (no exact zero, same signs: must mesh the same quad count).
    int zero_pillar = 0;
    static constexpr float ZERO_PILLAR_X = -52.75f;
    static constexpr float ZERO_PILLAR_Z = -110.0f;
    static constexpr float ZERO_PILLAR_HALF = 4.0f;
    static constexpr float ZERO_PILLAR_TOP_M = 416.0f;
    float zero_pillar_top = 0.0f; // metres, set by configure()

    void configure(const ChunkParam &p_param) override {
        SDFMetreBridge::configure(p_param);
        // The metre value the samples of that row get, op for op (ChunkMath::legacy_metres).
        WorldPos top;
        top.voxel = p_param.origin_voxel() + VoxelPos(0, int64_t(std::round(ZERO_PILLAR_TOP_M / p_param.voxel_size / 64.0f)) * 64, 0);
        top.frac = Float3{ 0.0f, 0.0f, 0.0f };
        zero_pillar_top = ChunkMath::legacy_metres(top, p_param).y + (zero_pillar == 2 ? 0.01f : 0.0f);
    }

    float zero_pillar_distance(const Vector3 &p) const {
        return MAX(p.y - zero_pillar_top,
                MAX(std::abs(p.x - ZERO_PILLAR_X) - ZERO_PILLAR_HALF, std::abs(p.z - ZERO_PILLAR_Z) - ZERO_PILLAR_HALF));
    }

    SDFDummy() = default;
    ~SDFDummy() override = default;

    float evaluate_metres(const Vector3 &p_world_pos) const override {
        // Mountain with sparse large caverns
        static const SimplexNoise::NoiseParam mountain_param{ 0.0035f, 1.0f, 2.0f, 0.5f };
        static const SimplexNoise::NoiseParam cave_blob_param{ 0.0080f, 1.0f, 2.0f, 0.5f }; // Low freq for big blobs

        const float x = p_world_pos.x;
        const float y = p_world_pos.y;
        const float z = p_world_pos.z;

        // Get height with noise
        const float n_height = SimplexNoise::fractal(5, mountain_param, Vector3(x, 0.0f, z)).w;
        const float mountain_height = 300.0f + n_height * 50.0f;

        // SDF = y - height (negative below surface)
        const float terrain_sdf = y - mountain_height;

        // Low-frequency cave blob selector (sparse big voids)
        const float cave_blob = SimplexNoise::fractal(3, cave_blob_param, Vector3(x, y, z)).w;

        // Depth from surface (positive = below terrain)
        const float depth = mountain_height - y;

        // Only carve caves in blob regions and when deep enough
        const float blob_mask = (cave_blob + 0.3f) * 0.5f; // Shift and scale to 0-0.8 range
        const float depth_fade = depth / 40.0f;
        const float cave_threshold = 0.5f + blob_mask * 0.4f + (1.0f - depth_fade) * 0.1f;

        // Simple cave field: just the blob
        const float cave_sdf = cave_threshold - cave_blob;

        // Carve caves: max(terrain, -caves)
        const float carved = MAX(terrain_sdf, -cave_sdf);

        float d = test_rig ? MIN(carved, SDFTestRig::evaluate(p_world_pos)) : carved;
        return zero_pillar ? MIN(d, zero_pillar_distance(p_world_pos)) : d;
    }

    //GLSL implementation
    String get_metre_glsl_code() const override {
        String code = R"glsl(
        vec4 legacy_terrain_sdf(vec3 pos){
            NoiseParam mountain_param;
            mountain_param.frequency = 0.0035;
            mountain_param.amplitude = 1.0;
            mountain_param.lacunarity = 2.0;
            mountain_param.persistence = 0.5;

            NoiseParam cave_blob_param;
            cave_blob_param.frequency = 0.0080;
            cave_blob_param.amplitude = 1.0;
            cave_blob_param.lacunarity = 2.0;
            cave_blob_param.persistence = 0.5;

            // -----------------------------------------
            // 1. Terrain SDF & Gradient
            // -----------------------------------------
            vec4 height_noise = fractal(5, mountain_param, vec3(pos.x, 0.0, pos.z));
            float base_height = 300.0;
            float mountain_height = base_height + height_noise.w * 50.0;
            
            // Calculate UNNORMALIZED terrain gradient & distance
            vec3 grad_height = height_noise.xyz * 50.0;
            vec3 unnorm_terrain_grad = vec3(-grad_height.x, 1.0, -grad_height.z);
            float unnorm_terrain_dist = pos.y - mountain_height;
            
            // Convert Terrain to true Euclidean Distance (Meters)
            float t_len = length(unnorm_terrain_grad);
            vec3 terrain_grad = unnorm_terrain_grad / t_len;
            float terrain_dist = unnorm_terrain_dist / t_len;
            
            vec4 terrain_val = vec4(terrain_grad, terrain_dist);
            
            // -----------------------------------------
            // 2. Cave Carving SDF & Gradient
            // -----------------------------------------
            vec4 cave_blob_noise = fractal(3, cave_blob_param, vec3(pos.x, pos.y, pos.z));
            float depth = mountain_height - pos.y; 
            
            float blob_mask = (cave_blob_noise.w + 0.3) * 0.5;
            float depth_fade = depth / 40.0;
            float cave_threshold = 0.5 + blob_mask * 0.4 + (1.0 - depth_fade) * 0.1;
            
            // Calculate UNNORMALIZED cave gradient & distance
            float unnorm_cave_dist = cave_threshold - cave_blob_noise.w;
            float unnorm_carve_dist = -unnorm_cave_dist; // Invert to carve out space
            
            // Using the unnormalized terrain gradient for accurate Chain Rule math
            vec3 unnorm_carve_grad = 0.8 * cave_blob_noise.xyz - 0.0025 * unnorm_terrain_grad;
            
            // Convert Cave to true Euclidean Distance (Meters)
            float c_len = length(unnorm_carve_grad) + 0.0001;
            vec3 carve_grad = unnorm_carve_grad / c_len;
            float carve_dist = unnorm_carve_dist / c_len;
            
            vec4 carve_val = vec4(carve_grad, carve_dist);
            
            // -----------------------------------------
            // 3. Smooth CSG Subtraction
            // -----------------------------------------
            // 5.0 determines the smoothness/bevel radius in meters.
            // You can increase this to 10.0 for massive, swooping cave entrances,
            // or lower it to 1.0 for harder, sharper rock edges.
            float blend_intensity = 5.0; 
            
            return sdf_softmax(terrain_val, carve_val, blend_intensity);
        }
        )glsl";
        String result = "sdf_softmax(terrain_val, carve_val, blend_intensity)";
        if (test_rig) {
            result = "sdf_min(" + result + ", test_rig_sdf(pos))";
            code = String(SDFTestRig::GLSL) + code;
        }
        if (zero_pillar) {
            result = "sdf_min(" + result + ", zero_pillar_sdf(pos))";
            // The top as raw float bits: the GPU compares against the exact same float.
            union {
                float f;
                uint32_t u;
            } bits{ zero_pillar_top };
            code = String(R"glsl(
        vec4 zero_pillar_sdf(vec3 pos) {
            vec4 d = vec4(0.0, 1.0, 0.0, pos.y - uintBitsToFloat(TOPu));
            float dx = abs(pos.x - (PX)) - HALF;
            float dz = abs(pos.z - (PZ)) - HALF;
            if (dx > d.w) d = vec4(sign(pos.x - (PX)), 0.0, 0.0, dx);
            if (dz > d.w) d = vec4(0.0, 0.0, sign(pos.z - (PZ)), dz);
            return d;
        }
        )glsl")
                            .replace("TOP", String::num_uint64(bits.u))
                            .replace("PX", String::num(ZERO_PILLAR_X, 9))
                            .replace("PZ", String::num(ZERO_PILLAR_Z, 9))
                            .replace("HALF", String::num(ZERO_PILLAR_HALF, 9)) +
                    code;
        }
        return code.replace("return sdf_softmax(terrain_val, carve_val, blend_intensity);", "return " + result + ";");
    }
};
