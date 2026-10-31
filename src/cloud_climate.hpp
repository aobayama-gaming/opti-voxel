#pragma once

#include "base_surface.hpp"
#include "climate.hpp"
#include "world_coord.hpp"

#include "ov/containers.hpp"
#include "ov/executor.hpp"

#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// The clouds' weather from a climate (docs/look.md): each biome's cover and cloud type (Climate::Biome::clouds) blended
// by its weight, on a coarse 3D lattice round the camera that the cloud passes read trilinearly (Clouds' map; 3D, as a
// curved base would tear a 2D map). A texel is linear in the weights, so filtering blends the biomes: (sum w_b cover_b,
// sum w_b type_b, sum w_b) over the biomes with clouds; where they set none (weight 1 - B), the passes take the sky's
// cover and the noise's type.
namespace CloudClimate {

struct Texel {
    float cover = 0.0f;
    float type = 0.0f;
    float weight = 0.0f; // of the biomes with clouds
};

// Whether some biome of p_table sets its clouds.
bool any(const Climate::Table &p_table);
// The texel of biome weights p_w (values in .w).
Texel of_weights(const Climate::Table &p_table, const Climate::Float4 p_w[Climate::MAX_BIOMES]);
// The texel of the climate at LOD0 voxel p_voxel, read on Λ_m (min(p_m, the spec's top); every field carried there:
// Climate::fields_at's): the biome weights of the climate low-passed at Λ_m's spacing.
Texel at(const Climate::Table &p_table, const Cascade::Spec &p_spec, const VoxelPos &p_voxel, int p_m);

// The lattice: SIZE^3 cells of 2^cell_shift LOD0 voxels, centred on a cell (Stream: the camera's). Only cells within
// MARGIN cells of the layer (every cell a trilinear read in it touches) are evaluated, each from the climate of the base
// point under its centre (a biome's clouds over it), read at the corner of the cell holding that point (one sample per
// such base cell, which a column's cells share); the others stay 0.
class Map {
public:
    static constexpr int SIZE = 64;
    static constexpr int MARGIN = 2;
    // The texel at a base cell's corner voxel, read on Λ_m (at): the map passes its cell shift.
    using SampleFn = std::function<Texel(const VoxelPos &p_voxel, int p_m)>;

    // A layer from p_bottom_m to p_top_m above p_base, LOD0 voxels of p_voxel_m.
    Map(const BaseSurface &p_base, int p_cell_shift, double p_bottom_m, double p_top_m, double p_voxel_m);

    int cell_shift() const { return shift; }
    Vector3i64 cell_of(const VoxelPos &p_voxel) const;
    // The cell a map for a camera at LOD0 voxel p_camera is centred on: the base point's under it (the layer stays in
    // the map from any height, the planet seen from orbit too).
    Vector3i64 centre_for(const VoxelPos &p_camera) const;
    // Whether the map centred on cell p_centre still serves a camera in cell p_cell.
    static bool covers(const Vector3i64 &p_centre, const Vector3i64 &p_cell);
    // The LOD0 voxel of the corner of the map centred on p_centre (texel 0's low corner).
    VoxelPos corner(const Vector3i64 &p_centre) const;
    // The map centred on p_centre as RGBA half floats (x fastest, then y, z: a 3D texture's texels), its largest cover
    // in r_max_cover. p_sample runs once per base cell the last build didn't evaluate (the others are its).
    std::vector<uint16_t> build(const Vector3i64 &p_centre, const SampleFn &p_sample, float &r_max_cover);

private:
    BaseSurface base;
    int shift;
    double low_vox; // the evaluated altitudes, LOD0 voxels
    double high_vox;
    ov::HashMap<Vector3i64, Texel, Vector3i64Hasher> cache; // by base cell, the last build's
};

// The map kept round a camera: built by a worker of its own (ov::executor's start_worker: a pool task waited behind a
// million trees' load, 29 s) when the camera is more than SIZE / 8 cells from the live map's centre (or there is none),
// one at a time; the host uploads what update hands back.
class Stream : public ov::WorkerLoop {
public:
    Stream(const Map &p_map, Map::SampleFn p_sample);
    ~Stream() override;
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;

    // The camera at LOD0 voxel p_camera: a finished build becomes the live map (true, its texels in r_texels: upload them
    // before drawing with corner / extent_vox), then a build is asked for if the camera left the live map.
    bool update(const VoxelPos &p_camera, std::vector<uint16_t> &r_texels);
    // Waits for the build asked for, if any (the next update collects it).
    void finish();
    // Whether a live map holds the camera of the last update (a teleport leaves it out until its build lands).
    bool live() const { return holds_camera; }
    // The live map's build (Trace::flow "climate_map": the host's upload ends it).
    uint64_t live_id() const { return live_build; }
    // The live map's corner (LOD0 voxel), its side (LOD0 voxels) and its largest cover.
    VoxelPos corner() const { return map.corner(live_centre); }
    double extent_vox() const { return std::ldexp(double(Map::SIZE), map.cell_shift()); }
    float max_cover() const { return live_max_cover; }

    // The worker's: builds the map asked for.
    bool step() override;

private:
    Map map; // its cache is the worker's; update reads only its cell size
    Map::SampleFn sample;
    std::unique_ptr<ov::Worker> worker;
    std::mutex mutex;
    std::condition_variable built;
    // Under mutex: a build asked for (busy) round next_centre, done (ready) and not collected yet.
    bool busy = false;
    bool ready = false;
    Vector3i64 next_centre;
    uint64_t next_id = 0;
    std::vector<uint16_t> next_texels;
    float next_max_cover = 0.0f;
    // The caller's.
    uint64_t asked = 0;
    bool has_live = false;
    bool holds_camera = false;
    Vector3i64 live_centre;
    float live_max_cover = 0.0f;
    uint64_t live_build = 0;
};

} // namespace CloudClimate
