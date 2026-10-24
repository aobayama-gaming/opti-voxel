#include "test_common.hpp"

#include "field_gpu.hpp"
#include "sdf_cascade.hpp"

#include <chrono>
#include <memory>
#include <thread>

// Without a local device (the headless unit tests) the GPU field ends FAILED and answers nothing: the table's batched
// field, attached or not, is the CPU's ground_field bit for bit (the site walks fall back).
TEST_CASE("FieldGPU: without a device it fails, and the table's batched field is the CPU's") {
    SDFCascade sdf;
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 18);
    std::shared_ptr<FieldGPU> gpu = std::make_shared<FieldGPU>();
    gpu->start(&sdf, param, 16);
    for (int i = 0; i < 2000 && gpu->state() == FieldGPU::STARTING; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(gpu->state() == FieldGPU::FAILED);
    const VoxelPos voxels[3] = { VoxelPos(10, 6000, -20), VoxelPos(-777, 6100, 31), VoxelPos(1 << 20, 5900, 5) };
    const int lods[3] = { 0, 5, 3 };
    ExactNoise::Float4 out[3];
    CHECK(!gpu->ground_field(voxels, lods, 3, out));
    sdf.attach_field_gpu(gpu); // attached but failed: the CPU's
    sdf.ground_field_batch(voxels, lods, 3, out);
    for (int i = 0; i < 3; i++) {
        CHECK(ExactNoise::same_bits(out[i], sdf.ground_field(voxels[i], lods[i])));
    }
    sdf.attach_field_gpu(nullptr);
    gpu->stop();
    CHECK(gpu->state() == FieldGPU::FAILED);
}
