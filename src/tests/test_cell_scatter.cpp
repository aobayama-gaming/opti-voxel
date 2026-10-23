#include "test_common.hpp"

#include "cell_scatter.hpp"
#include "sdf_cascade.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

using namespace Archetype;

namespace {

// Rolling hills along x, as a field in LOD0 voxels (the same at every LOD).
FieldFn hills_field(const VoxelPos &p_origin) {
    return [p_origin](const VoxelPos &p_v, int) {
        const double tau = 6.283185307179586, amp = 40.0, wave = 900.0;
        const double x = double(p_v.x - p_origin.x);
        const double w = double(p_v.y - p_origin.y) - amp * std::sin(tau * x / wave);
        return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave)), 1.0f, 0.0f, float(w) };
    };
}

// Two spheres per site, sized by its seed, the second one up the axis.
void two_spheres(const Site &p_site, const UpAxis &p_up, int, std::vector<TableRecord> &r_out) {
    for (int i = 0; i < 2; i++) {
        TableRecord rec;
        rec.def = 0;
        rec.op = Combine::SMOOTH_UNION;
        rec.set_k(2.0f);
        rec.anchor = p_site.base + VoxelPos(std::llround(10.0 * i * p_up[0]), std::llround(10.0 * i * p_up[1]), std::llround(10.0 * i * p_up[2]));
        rec.params[0] = 4.0f + float(p_site.seed % 5u);
        rec.salt = p_site.seed + uint32_t(i);
        r_out.push_back(rec);
    }
}

bool same_record(const TableRecord &p_a, const TableRecord &p_b) {
    return p_a.def == p_b.def && p_a.op == p_b.op && p_a.anchor == p_b.anchor && std::memcmp(p_a.params, p_b.params, sizeof(p_a.params)) == 0 &&
            std::memcmp(p_a.rotation, p_b.rotation, sizeof(p_a.rotation)) == 0 && p_a.salt == p_b.salt;
}

} // namespace

TEST_CASE("Cell scatter: two halves of a box give the whole, sites and records; accept leaves sites out") {
    const VoxelPos origin(int64_t(1) << 36, 2000, -(int64_t(1) << 35));
    CellScatter s;
    s.salt = 0xABCDu;
    s.side = 60;
    s.zone = VoxelBox{ origin - VoxelPos(1500, 600, 1500), origin + VoxelPos(1500, 600, 1500), true };
    s.ground = hills_field(origin);
    s.up = [](const VoxelPos &) { return UP_Y; };
    s.generate = two_spheres;
    CHECK(s.valid());
    if (!(s.valid())) {
        return;
    }
    // A query wider than the zone: clipped to it.
    std::vector<CellSite> whole, a, b;
    s.sites(s.zone.grown(5000), whole);
    const int64_t mid = origin.x + 137;
    VoxelBox left = s.zone, right = s.zone;
    left.hi.x = mid;
    right.lo.x = mid + 1;
    s.sites(left, a);
    s.sites(right, b);
    std::vector<CellSite> halves = a;
    halves.insert(halves.end(), b.begin(), b.end());
    std::sort(halves.begin(), halves.end(), [](const CellSite &p_x, const CellSite &p_y) { return cell_order(p_x.cell, p_y.cell); });
    MESSAGE("whole ", whole.size(), " sites, halves ", a.size(), " + ", b.size());
    CHECK(whole.size() > 1000);
    CHECK(halves.size() == whole.size());
    if (!(halves.size() == whole.size())) {
        return;
    }
    for (size_t i = 0; i < whole.size(); i++) {
        CHECK(halves[i].cell == whole[i].cell);
        CHECK(halves[i].site.base == whole[i].site.base);
        CHECK(halves[i].site.seed == whole[i].site.seed);
        CHECK(halves[i].site.variant == whole[i].site.variant);
        CHECK(s.zone.contains(whole[i].site.base));
        std::vector<TableRecord> x, y;
        s.grow(whole[i].site, -1, x);
        s.grow(halves[i].site, -1, y);
        CHECK(x.size() == y.size());
        if (!(x.size() == y.size())) {
            return;
        }
        for (size_t j = 0; j < x.size(); j++) {
            CHECK(same_record(x[j], y[j]));
        }
    }
    // Accept: the sites left out are exactly the refused ones.
    s.accept = [mid](const VoxelPos &p_v) { return p_v.x > mid; };
    std::vector<CellSite> accepted;
    s.sites(s.zone, accepted);
    CHECK(accepted.size() == b.size());
}

// The profile is read from samples, the sources grow other sites: every record they grow must still be within it (the
// keys of a streamed source trust it), on the forest and the rock scatter at 5 cm.
TEST_CASE("Cell scatter: the profile bounds every record of the forest and the scatter") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.set_forest(200, 80.0);
    sdf.set_scatter(400, 80.0);
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 18));
    sdf.configure(param, 18);
    CHECK(sdf.get_sources().size() == 2);
    if (!(sdf.get_sources().size() == 2)) {
        return;
    }
    for (const SDFTable::Source &source : sdf.get_sources()) {
        const std::string name = source.name;
        CAPTURE(name);
        const SourceProfile profile(sdf.get_library(), source.scatter, sdf.record_finish(), MomentLayerParams(), 18);
        std::vector<CellSite> sites;
        source.scatter.sites(source.scatter.zone, sites);
        int64_t records = 0, outside = 0;
        std::vector<TableRecord> grown;
        for (const CellSite &cs : sites) {
            grown.clear();
            source.scatter.grow(cs.site, -1, grown);
            for (TableRecord &rec : grown) {
                finish_table_record(sdf.get_library(), sdf.record_finish(), rec);
                outside += profile.covers(cs.site.base, rec) ? 0 : 1;
                records++;
            }
        }
        const double mean = sites.empty() ? 0.0 : double(records) / double(sites.size());
        MESSAGE(name, ": ", sites.size(), " sites, ", records, " records (", mean, " per site, profile ", profile.records_per_site, "), reach ",
                profile.reach * double(param.voxel_size), " m, lod_end ", profile.lod_end, ", k_max ", profile.k_max, ", ", profile.joins.size(), " joined materials");
        CHECK(sites.size() > 100);
        CHECK(outside == 0);
        CHECK(profile.lod_end > 0);
        CHECK(profile.lod_end <= 18);
        CHECK(std::fabs(mean - profile.records_per_site) <= 0.15 * profile.records_per_site);
        // What a site has lost grows with the LOD, and from its last LOD on all of it is gone.
        double before = 0.0;
        for (int l = 0; l < 18; l++) {
            double m = 0.0;
            for (const std::pair<uint32_t, Moments> &p : profile.lost[l].moments.parts) {
                m += p.second.m0;
            }
            CHECK(m >= before * (1.0 - 1e-9));
            before = m;
        }
        CHECK(before > 0.0);
    }
}
