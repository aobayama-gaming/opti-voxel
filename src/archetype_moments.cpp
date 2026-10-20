// The archetype records' moments (archetype_table.hpp record_moments, span_moments; the math in instance_moments.hpp).
#include "archetype_table.hpp"

namespace Archetype {

Moments record_moments(const TableDef &p_def, const TableRecord &p_record) {
    if (p_def.moments) {
        return p_def.moments(p_record.params).rotated(p_record.rotation);
    }
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    const double size[3] = { double(hi.x) - double(lo.x), double(hi.y) - double(lo.y), double(hi.z) - double(lo.z) };
    const double cell = size[0] * size[1] * size[2] / double(FALLBACK_GRID * FALLBACK_GRID * FALLBACK_GRID);
    const DefRecord dr = p_record.def_record();
    std::vector<Float4> regs;
    MomentSums sums;
    for (int i = 0; i < FALLBACK_GRID * FALLBACK_GRID * FALLBACK_GRID; i++) {
        const int g[3] = { i % FALLBACK_GRID, i / FALLBACK_GRID % FALLBACK_GRID, i / (FALLBACK_GRID * FALLBACK_GRID) };
        const double l[3] = { double(lo.x) + (g[0] + 0.5) * size[0] / FALLBACK_GRID, double(lo.y) + (g[1] + 0.5) * size[1] / FALLBACK_GRID,
            double(lo.z) + (g[2] + 0.5) * size[2] / FALLBACK_GRID };
        double w[3];
        local_to_world(p_record.rotation, l, w);
        run_registers(p_def.program, WorldCoord::offset_pos(p_record.anchor, w), 0, regs, &dr);
        if (!(regs[p_def.program.output].w > 0.0f)) {
            sums.add(cell, w);
        }
    }
    return sums.moments();
}

MaterialMoments span_moments(const DefLibrary &p_library, const TableRecord *p_records, uint32_t p_count, const VoxelPos &p_anchor) {
    MaterialMoments out;
    for (uint32_t i = 0; i < p_count; i++) {
        const TableRecord &rec = p_records[i];
        if (rec.removed || is_cutter(rec.op) || rec.op == Combine::ACCUM) {
            continue;
        }
        const double offset[3] = { double(rec.anchor.x - p_anchor.x), double(rec.anchor.y - p_anchor.y), double(rec.anchor.z - p_anchor.z) };
        out.add(record_material(rec.material, p_library.material(rec.def), rec.op), record_moments(p_library[rec.def], rec), offset);
    }
    return out;
}

} // namespace Archetype
