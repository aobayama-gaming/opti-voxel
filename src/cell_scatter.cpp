// A procedural instance source (cell_scatter.hpp).
#include "cell_scatter.hpp"

#include "parallel_for.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <cmath>

namespace Archetype {

void CellScatter::sites(const VoxelBox &p_query, std::vector<CellSite> &r_out, bool p_snap) const {
    const VoxelBox box = p_query.clipped(zone);
    if (!box.valid || !valid()) {
        return;
    }
    const size_t first = r_out.size();
    surface_sites(box.lo, box.hi, side, salt, ground, r_out, p_snap);
    if (accept) {
        r_out.erase(std::remove_if(r_out.begin() + first, r_out.end(), [this](const CellSite &p_s) { return !accept(p_s.site.base); }), r_out.end());
    }
}

namespace {

double largest_variance(const Moments &p_m) {
    double lambda[3];
    p_m.principal(lambda);
    return lambda[2];
}

} // namespace

SourceProfile::SourceProfile(const DefLibrary &p_library, const CellScatter &p_scatter, const RecordFinish &p_finish, const MomentLayerParams &p_layers,
        int p_lod_count) {
    ERR_FAIL_COND_MSG(!p_scatter.generate, "SourceProfile: no generate.");
    std::vector<std::vector<TableRecord>> parts(SAMPLES);
    std::vector<std::vector<MomentItem>> items(SAMPLES);
    parallel_for(uint32_t(SAMPLES), "SourceProfile", [&](uint32_t p_i) {
        const ExactNoise::UVec3 h = ExactNoise::hash3(uint32_t(p_i), p_scatter.salt, 0x9E0F11Eu);
        p_scatter.generate(Site{ VoxelPos(), h.x, h.y >> 16 }, UP_Y, parts[p_i]);
        for (TableRecord &rec : parts[p_i]) {
            finish_table_record(p_library, p_finish, rec);
            MomentItem item = record_item(p_library, rec);
            prepare_moment_item(item);
            items[p_i].push_back(item);
        }
    });
    // In sample order: the same bits however the pool ran.
    const double share = 1.0 / double(SAMPLES);
    lost.assign(p_lod_count, Lost());
    std::vector<MomentItem> all;
    size_t records = 0;
    for (int i = 0; i < SAMPLES; i++) {
        records += parts[i].size();
        for (const TableRecord &rec : parts[i]) {
            // A record's shape lies in its box, so within the box's farthest corner of the base (the origin) under any
            // turn of the site: a turned site's boxes stay within that distance on every axis.
            for (int corner = 0; corner < 8; corner++) {
                double d2 = 0.0;
                for (int a = 0; a < 3; a++) {
                    const double v = double(rec.anchor[a] + ((corner >> a) & 1 ? rec.box_hi[a] : rec.box_lo[a]));
                    d2 += v * v;
                }
                reach = MAX(reach, std::sqrt(d2));
            }
            lod_end = MAX(lod_end, rec.max_lod == TableRecord::NEVER_DROPPED ? p_lod_count : int(rec.max_lod));
            k_max = MAX(k_max, rec.k);
            margin_max = MAX(margin_max, rec.margin_c);
        }
        for (const MomentItem &item : items[i]) {
            if (item.moments.empty()) {
                continue;
            }
            part_variance = MAX(part_variance, largest_variance(item.moments));
            const double offset[3] = { double(item.anchor.x), double(item.anchor.y), double(item.anchor.z) }; // from the base
            for (int l = 0; l < p_lod_count; l++) {
                const double w = share * item.lost_share(l);
                if (!(w > 0.0)) {
                    continue;
                }
                Lost &at = lost[l];
                at.moments.add(item.material, item.moments.scaled(w), offset);
                std::array<double, 6> &area = at.area[item.material];
                for (int k = 0; k < 6; k++) {
                    area[k] += w * item.area[k];
                }
            }
            all.push_back(item);
        }
    }
    records_per_site = double(records) * share;
    reach *= REACH_SAFETY;
    k_max = float(double(k_max) * REACH_SAFETY);
    margin_max = float(double(margin_max) * REACH_SAFETY); // a rock's grows with its size
    lod_end = MIN(lod_end, p_lod_count);
    MomentLayerParams params = p_layers;
    params.lod_min = 0;
    params.lod_max = p_lod_count - 1;
    joins = moment_joins(all, params);
    for (Lost &at : lost) {
        for (const std::pair<uint32_t, Moments> &m : at.moments.parts) {
            at.variance = MAX(at.variance, largest_variance(m.second));
        }
    }
}

bool SourceProfile::covers(const VoxelPos &p_base, const TableRecord &p_record) const {
    const VoxelBox reach_box = VoxelBox{ p_base, p_base, true }.grown(int64_t(std::floor(reach)));
    const VoxelBox b = record_box(p_record);
    const int listed_end = p_record.max_lod == TableRecord::NEVER_DROPPED ? int(lost.size()) : int(p_record.max_lod);
    return reach_box.contains(b.lo) && reach_box.contains(b.hi) && listed_end <= lod_end && p_record.k <= k_max && p_record.margin_c <= margin_max;
}

} // namespace Archetype
