#include "cpu_mesher.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ov::dawn {

namespace {

// The 12 edges of a cube, as corner pairs (corner c = x + 2y + 4z).
constexpr int EDGES[12][2] = {
    { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, // x
    { 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 }, // y
    { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, // z
};

int corner_x(int p_c) { return p_c & 1; }
int corner_y(int p_c) { return (p_c >> 1) & 1; }
int corner_z(int p_c) { return (p_c >> 2) & 1; }

} // namespace

MeshData mesh_surface_nets(const std::function<float(const VoxelPos &)> &p_sample, const VoxelPos &p_min, int p_cells, float p_voxel_size, int p_lod) {
    MeshData mesh;
    // One cell of overlap on every side: a quad on the chunk border needs the four cells around its edge, two of
    // which live in the neighbouring chunk. Sampling them here closes the seam (the neighbour does not emit that
    // edge's quad, so no triangle is drawn twice).
    constexpr int BORDER = 1;
    const int stride = p_cells + 2 * BORDER + 1; // samples -1 .. p_cells + 1
    const int cells_n = p_cells + 2 * BORDER; // cells -1 .. p_cells
    const int64_t step = int64_t(1) << p_lod;

    std::vector<float> samples(size_t(stride) * stride * stride);
    std::vector<int32_t> cell_vertex(size_t(cells_n) * cells_n * cells_n, -1);

    auto sample_at = [&](int p_x, int p_y, int p_z) -> float & {
        const int x = p_x + BORDER;
        const int y = p_y + BORDER;
        const int z = p_z + BORDER;
        return samples[size_t(z * stride * stride + y * stride + x)];
    };
    for (int z = -BORDER; z <= p_cells + BORDER; z++) {
        for (int y = -BORDER; y <= p_cells + BORDER; y++) {
            for (int x = -BORDER; x <= p_cells + BORDER; x++) {
                sample_at(x, y, z) = p_sample(VoxelPos(p_min.x + int64_t(x) * step, p_min.y + int64_t(y) * step, p_min.z + int64_t(z) * step));
            }
        }
    }

    auto cell_index = [&](int p_x, int p_y, int p_z) -> int32_t & {
        const int x = p_x + BORDER;
        const int y = p_y + BORDER;
        const int z = p_z + BORDER;
        return cell_vertex[size_t(z * cells_n * cells_n + y * cells_n + x)];
    };

    // One vertex per sign-changing cell: the average of its edge crossings.
    for (int z = -BORDER; z <= p_cells; z++) {
        for (int y = -BORDER; y <= p_cells; y++) {
            for (int x = -BORDER; x <= p_cells; x++) {
                float corner[8];
                for (int c = 0; c < 8; c++) {
                    corner[c] = sample_at(x + corner_x(c), y + corner_y(c), z + corner_z(c));
                }
                bool any_negative = false;
                bool any_positive = false;
                for (float v : corner) {
                    any_negative |= v < 0.0f;
                    any_positive |= v >= 0.0f;
                }
                if (!any_negative || !any_positive) {
                    continue;
                }
                Float3 sum{ 0.0f, 0.0f, 0.0f };
                int crossings = 0;
                for (const auto &edge : EDGES) {
                    const float a = corner[edge[0]];
                    const float b = corner[edge[1]];
                    if ((a < 0.0f) == (b < 0.0f)) {
                        continue;
                    }
                    const float t = a / (a - b);
                    const Float3 pa{ float(corner_x(edge[0])), float(corner_y(edge[0])), float(corner_z(edge[0])) };
                    const Float3 pb{ float(corner_x(edge[1])), float(corner_y(edge[1])), float(corner_z(edge[1])) };
                    sum.x += pa.x + (pb.x - pa.x) * t;
                    sum.y += pa.y + (pb.y - pa.y) * t;
                    sum.z += pa.z + (pb.z - pa.z) * t;
                    crossings++;
                }
                if (crossings == 0) {
                    continue;
                }
                const Float3 local{ float(x) + sum.x / float(crossings), float(y) + sum.y / float(crossings), float(z) + sum.z / float(crossings) };
                const VoxelPos voxel(p_min.x + int64_t(local.x * float(step)), p_min.y + int64_t(local.y * float(step)), p_min.z + int64_t(local.z * float(step)));
                const float gx = p_sample(VoxelPos(voxel.x + step, voxel.y, voxel.z)) - p_sample(VoxelPos(voxel.x - step, voxel.y, voxel.z));
                const float gy = p_sample(VoxelPos(voxel.x, voxel.y + step, voxel.z)) - p_sample(VoxelPos(voxel.x, voxel.y - step, voxel.z));
                const float gz = p_sample(VoxelPos(voxel.x, voxel.y, voxel.z + step)) - p_sample(VoxelPos(voxel.x, voxel.y, voxel.z - step));
                const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
                Float3 normal{ 0.0f, 1.0f, 0.0f };
                if (len > 1e-9f) {
                    normal = Float3{ gx / len, gy / len, gz / len };
                }
                const Float3 world{
                    (float(p_min.x) + local.x * float(step)) * p_voxel_size,
                    (float(p_min.y) + local.y * float(step)) * p_voxel_size,
                    (float(p_min.z) + local.z * float(step)) * p_voxel_size,
                };
                cell_index(x, y, z) = int32_t(mesh.vertices.size());
                mesh.vertices.push_back(MeshVertex{ world, normal });
            }
        }
    }

    // One quad per sign-changing edge, from the four cells around it. Every edge in [0, p_cells) has its four
    // cells in [-1, p_cells], the border included.
    auto quad = [&](int32_t p0, int32_t p1, int32_t p2, int32_t p3, bool p_flip) {
        if (p0 < 0 || p1 < 0 || p2 < 0 || p3 < 0) {
            return;
        }
        if (p_flip) {
            mesh.indices.insert(mesh.indices.end(), { uint32_t(p0), uint32_t(p1), uint32_t(p3), uint32_t(p0), uint32_t(p3), uint32_t(p2) });
        } else {
            mesh.indices.insert(mesh.indices.end(), { uint32_t(p0), uint32_t(p3), uint32_t(p1), uint32_t(p0), uint32_t(p2), uint32_t(p3) });
        }
    };

    for (int z = 0; z < p_cells; z++) {
        for (int y = 0; y < p_cells; y++) {
            for (int x = 0; x < p_cells; x++) {
                const bool solid = sample_at(x, y, z) < 0.0f;
                if (solid != (sample_at(x + 1, y, z) < 0.0f)) {
                    quad(cell_index(x, y, z), cell_index(x, y - 1, z), cell_index(x, y, z - 1), cell_index(x, y - 1, z - 1), solid);
                }
                if (solid != (sample_at(x, y + 1, z) < 0.0f)) {
                    quad(cell_index(x, y, z), cell_index(x - 1, y, z), cell_index(x, y, z - 1), cell_index(x - 1, y, z - 1), solid);
                }
                if (solid != (sample_at(x, y, z + 1) < 0.0f)) {
                    quad(cell_index(x, y, z), cell_index(x - 1, y, z), cell_index(x, y - 1, z), cell_index(x - 1, y - 1, z), solid);
                }
            }
        }
    }
    return mesh;
}

} // namespace ov::dawn
