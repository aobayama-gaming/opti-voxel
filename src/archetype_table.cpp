// The archetype table: the library, the records, the locality rule and the fold (archetype_table.hpp).
#include "archetype_table.hpp"
#include "chunk_palette.hpp"
#include "sdf_base.hpp"

#include "ov/bytes.hpp"
#include "ov/format.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"
#include "ov/str.hpp"

#include <cmath>
#include <cstring>

namespace Archetype {

using ExactNoise::exact_pow2;
using ExactNoise::glsl_float;

int DefLibrary::add(const TableDefPtr &p_def, uint32_t p_material) {
    list.push_back(p_def);
    materials.push_back(p_material);
    return int(list.size() - 1);
}

namespace {

// Half the smallest side of the band depth's box (voxels).
double band_radius(const TableDef &p_def, const TableRecord &p_record) {
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    double side = std::min(std::min(double(hi.x) - double(lo.x), double(hi.y) - double(lo.y)), double(hi.z) - double(lo.z)) + 2.0 * rotation_margin(lo, hi);
    for (int a = 0; a < 3; a++) {
        side = std::min(side, double(p_record.box_hi[a]) - double(p_record.box_lo[a]));
    }
    return 0.5 * side;
}

} // namespace

double band_depth(const TableDef &p_def, const TableRecord &p_record) {
    return band_radius(p_def, p_record) * double(p_record.lipschitz) + double(p_record.k);
}

void set_band(TableRecord &r_record, const TableDef &p_def, bool p_enabled, const ErodeRule &p_rule) {
    r_record.max_lod = TableRecord::NEVER_DROPPED;
    r_record.erode = 0.0f;
    r_record.erode_from = 1.0f;
    if (!p_enabled) {
        return;
    }
    r_record.erode = p_rule.rate;
    const double rate = p_rule.rate;
    const double depth = band_depth(p_def, r_record);
    // The first LOD erosion from LOD 0 can make the part vanish by, then later while the grid still resolves it.
    int drop = 0;
    while (drop < 31 && rate * (std::ldexp(1.0, drop) - 1.0) < depth) {
        drop++;
    }
    if (p_rule.resolve > 0.0f) {
        const double half = record_moments(p_def, r_record).half_thickness();
        while (drop < 31 && double(p_rule.resolve) * std::ldexp(1.0, drop) <= half) {
            drop++;
        }
    }
    if (rate * (std::ldexp(1.0, drop) - 1.0) < depth) {
        return; // deeper than 31 LODs can erode: never dropped (erode_from stays 1)
    }
    // The latest onset from which the erosion still reaches the depth at the drop (onset 0 always does).
    int onset = ov::max(drop - 1, 0);
    while (onset > 0 && rate * (std::ldexp(1.0, drop) - std::ldexp(1.0, onset)) < depth) {
        onset--;
    }
    r_record.erode_from = exact_pow2(onset);
    r_record.max_lod = uint8_t(drop);
}

void finish_record(const TableDef &p_def, TableRecord &r_record, float p_skirt_depth, bool p_band, const ErodeRule &p_rule) {
    const Bounds wb = record_world_bounds(p_def, r_record);
    for (int a = 0; a < 3; a++) {
        r_record.box_lo[a] = int32_t(wb.lo[a] - r_record.anchor[a]);
        r_record.box_hi[a] = int32_t(wb.hi[a] - r_record.anchor[a]);
    }
    r_record.lipschitz = wb.lipschitz;
    r_record.margin_c = 2.0f * wb.lipschitz + p_skirt_depth;
    set_band(r_record, p_def, p_band, p_rule);
}

void encode_record(uint8_t *r_slot, const TableRecord &p_record) {
    using namespace WorldCoord;
    ov::put_u32(r_slot + 0, p_record.def);
    ov::put_u32(r_slot + 4, uint32_t(p_record.op));
    ov::put_f32(r_slot + 8, p_record.k);
    ov::put_f32(r_slot + 12, p_record.inv_k);
    ov::put_u32(r_slot + 16, lo32(p_record.anchor.x));
    ov::put_u32(r_slot + 20, lo32(p_record.anchor.y));
    ov::put_u32(r_slot + 24, lo32(p_record.anchor.z));
    ov::put_u32(r_slot + 28, p_record.salt);
    ov::put_s32(r_slot + 32, hi32(p_record.anchor.x));
    ov::put_s32(r_slot + 36, hi32(p_record.anchor.y));
    ov::put_s32(r_slot + 40, hi32(p_record.anchor.z));
    ov::put_f32(r_slot + 44, p_record.erode_from);
    for (int i = 0; i < 9; i++) {
        ov::put_f32(r_slot + 48 + (i / 3) * 16 + (i % 3) * 4, p_record.rotation[i]);
    }
    for (int i = 0; i < 8; i++) {
        ov::put_f32(r_slot + 96 + i * 4, p_record.params[i]);
    }
    for (int i = 0; i < 3; i++) {
        ov::put_s32(r_slot + 128 + i * 4, p_record.box_lo[i]);
        ov::put_s32(r_slot + 144 + i * 4, p_record.box_hi[i]);
    }
    ov::put_f32(r_slot + 140, p_record.margin_c);
    ov::put_f32(r_slot + 156, p_record.erode);
}

// ---- the fold ----

namespace {

// The dual-number sequence of Builder::smooth_min (archetype_build.cpp) with runtime operands: same
// operations, same order (the GPU's arch_smooth_min mirrors it).
Float4 smooth_min4(const Float4 &p_a, const Float4 &p_b, float p_k, float p_inv_k) {
    const Float4 diff{ p_a.x - p_b.x, p_a.y - p_b.y, p_a.z - p_b.z, p_a.w - p_b.w };
    const Float4 d = diff.w < 0.0f ? Float4{ -diff.x, -diff.y, -diff.z, -diff.w }
                                   : Float4{ diff.x, diff.y, diff.z, std::fabs(diff.w) };
    const Float4 t{ d.x * p_inv_k, d.y * p_inv_k, d.z * p_inv_k, d.w * p_inv_k };
    Float4 h{ -t.x, -t.y, -t.z, 1.0f - t.w };
    if (0.0f > h.w) {
        h = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    const Float4 m = p_b.w < p_a.w ? p_b : p_a;
    const Float4 h2{ h.x * h.w + h.w * h.x, h.y * h.w + h.w * h.y, h.z * h.w + h.w * h.z, h.w * h.w };
    const float c = 0.25f * p_k;
    const Float4 amt{ h2.x * c, h2.y * c, h2.z * c, h2.w * c };
    return Float4{ m.x - amt.x, m.y - amt.y, m.z - amt.z, m.w - amt.w };
}

Float4 smooth_max4(const Float4 &p_a, const Float4 &p_b, float p_k, float p_inv_k) {
    const Float4 na{ -p_a.x, -p_a.y, -p_a.z, -p_a.w };
    const Float4 nb{ -p_b.x, -p_b.y, -p_b.z, -p_b.w };
    const Float4 s = smooth_min4(na, nb, p_k, p_inv_k);
    return Float4{ -s.x, -s.y, -s.z, -s.w };
}

// The four Combine operators, on registers.
Float4 combine4(const Float4 &p_r, const Float4 &p_c, const TableRecord &p_record) {
    switch (p_record.op) {
        case Combine::UNION:
            return p_c.w < p_r.w ? p_c : p_r;
        case Combine::SMOOTH_UNION:
            return smooth_min4(p_r, p_c, p_record.k, p_record.inv_k);
        case Combine::SUBTRACT: {
            const Float4 n{ -p_c.x, -p_c.y, -p_c.z, -p_c.w };
            return n.w > p_r.w ? n : p_r;
        }
        default: { // SMOOTH_SUBTRACT
            const Float4 n{ -p_c.x, -p_c.y, -p_c.z, -p_c.w };
            return smooth_max4(p_r, n, p_record.k, p_record.inv_k);
        }
    }
}

} // namespace

// arch_record_wins tests op <= 1 for the unions.
static_assert(uint8_t(Combine::UNION) == 0 && uint8_t(Combine::SMOOTH_UNION) == 1, "the unions are ops 0 and 1");

bool record_wins(const Float4 &p_r, const Float4 &p_c, Combine p_op) {
    return p_op == Combine::UNION || p_op == Combine::SMOOTH_UNION ? p_c.w < p_r.w : -p_c.w > p_r.w;
}

uint32_t record_material(uint32_t p_material, uint32_t p_type_material, Combine p_op) {
    const uint32_t m = p_material == Material::FROM_TYPE ? p_type_material : p_material;
    return m == Material::INHERIT && !is_cutter(p_op) ? Material::GROUND : m;
}

Float4 fold_onto(const Float4 &p_r, const DefLibrary &p_library, RecordView p_records, const uint32_t *p_list, uint32_t p_count,
        const WorldPos &p_pos, int p_lod, bool p_box_test, bool *r_touched, const uint16_t *p_palette, uint32_t *r_material, bool *r_blobs) {
    thread_local std::vector<Float4> regs;
    Float4 r = p_r;
    Float4 phi{ 0.0f, 0.0f, 0.0f, 0.0f }; // the moment blobs' summed occupancy
    bool blobs = false;
    float best = 0.0f; // the strongest blob's occupancy and palette index
    uint32_t best_index = ChunkPalette::INHERIT_INDEX;
    for (uint32_t i = 0; i < p_count; i++) {
        const TableRecord &rec = p_records[p_list[i]];
        if (p_box_test && !in_influence_box(rec, p_pos, p_lod)) {
            continue;
        }
        const DefRecord dr = rec.def_record();
        run_registers(p_library[rec.def].program, p_pos, p_lod, regs, &dr);
        Float4 c = regs[p_library[rec.def].program.output];
        if (rec.op == Combine::ACCUM) {
            phi = Float4{ phi.x + c.x, phi.y + c.y, phi.z + c.z, phi.w + c.w };
            if (c.w > best) {
                best = c.w;
                best_index = p_palette ? p_palette[i] : ChunkPalette::INHERIT_INDEX;
            }
            blobs = true;
            if (r_touched) {
                *r_touched = true;
            }
            continue;
        }
        c.w = c.w + band_erosion(rec.erode, rec.erode_from, p_lod); // the band rule's erosion (set_band)
        if (r_material && p_palette && p_palette[i] != ChunkPalette::INHERIT_INDEX && record_wins(r, c, rec.op)) {
            *r_material = p_palette[i];
        }
        r = combine4(r, c, rec);
        if (r_touched) {
            *r_touched = true;
        }
    }
    if (r_blobs) {
        *r_blobs = *r_blobs || blobs;
    }
    return blobs ? moment_surface(r, phi, p_lod, best_index, r_material) : r;
}

Float4 moment_surface(const Float4 &p_r, const Float4 &p_phi, int p_lod, uint32_t p_best, uint32_t *r_material) {
    const float h = exact_pow2(p_lod);
    const Float4 lift{ p_r.x - p_phi.x * h, p_r.y - p_phi.y * h, p_r.z - p_phi.z * h, p_r.w - p_phi.w * h };
    const float n2 = p_phi.x * p_phi.x + p_phi.y * p_phi.y + p_phi.z * p_phi.z;
    const float gh = ExactNoise::det_rsqrt(ov::max(n2, MOMENT_GRAD_FLOOR)) * h;
    const Float4 occ{ -p_phi.x * gh, -p_phi.y * gh, -p_phi.z * gh, ov::max((MOMENT_ISO - p_phi.w) * gh, -(MOMENT_DEPTH_CELLS * h)) };
    if (r_material && p_best != ChunkPalette::INHERIT_INDEX && p_phi.w >= MOMENT_MATERIAL_ISO) {
        *r_material = p_best;
    }
    return occ.w < lift.w ? occ : lift;
}

Float4 finish_table(const Float4 &p_out, int p_lod, float p_out_scale, bool p_normalize) {
    const float v = p_out.w * p_out_scale;
    if (!p_normalize) {
        return Float4{ p_out.x, p_out.y, p_out.z, v };
    }
    const float n2 = p_out.x * p_out.x + p_out.y * p_out.y + p_out.z * p_out.z;
    const float s = exact_pow2(p_lod);
    const float gr = ExactNoise::det_rsqrt(std::max(n2, exact_pow2(GRAD_FLOOR_EXP) * s * s));
    return Float4{ p_out.x * gr, p_out.y * gr, p_out.z * gr, v };
}

// ---- GLSL ----

const char GLSL_DEF_RECORD_STRUCT[] = R"glsl(
struct DefRecord {
    uvec4 head;      // x: def, y: op, z: k bits, w: inv_k bits
    uvec4 anchor_lo; // xyz: low words, w: noise salt
    ivec4 anchor_hi; // xyz: high words, w: erode_from bits (2^onset, set_band)
    vec4 rot[3];     // world -> local, row major
    float params[8];
    ivec4 box_lo;    // xyz: influence box low (relative), w: margin_c bits
    ivec4 box_hi;    // xyz: influence box high, w: erode bits (voxels per cell past the onset, set_band)
};
)glsl";

// arch_margin / arch_meets_box / arch_in_box / arch_smooth_min / arch_smooth_max / arch_combine: the GPU mirror of
// influence_margin, in_influence_cube, in_influence_box and combine4. Spliced with the terrain (get_glsl_code), after the noise.
static const char GLSL_TABLE_FOLD[] = R"glsl(

int arch_margin(DefRecord rec, int lod) {
    precise float mf = min(uintBitsToFloat(rec.head.z) + uintBitsToFloat(rec.box_lo.w) * exact_pow2(lod), float(LIMIT));
    return int(ceil(mf)) + 2;
}

// in_influence_cube: whether the influence box may hold a point of [lo, hi].
bool arch_meets_box(WorldPos lo, WorldPos hi, DefRecord rec, int lod) {
    ivec3 dlo = sat_diff(lo, rec.anchor_lo.xyz, rec.anchor_hi.xyz, LIMIT);
    ivec3 dhi = sat_diff(hi, rec.anchor_lo.xyz, rec.anchor_hi.xyz, LIMIT);
    int m = arch_margin(rec, lod);
    return all(greaterThanEqual(dhi, rec.box_lo.xyz - ivec3(m))) && all(lessThanEqual(dlo, rec.box_hi.xyz + ivec3(m)));
}

bool arch_in_box(WorldPos p, DefRecord rec, int lod) {
    return arch_meets_box(p, p, rec, lod);
}

vec4 arch_smooth_min(vec4 a, vec4 b, float k, float inv_k) {
    precise vec4 diff = a - b;
    precise vec4 d = diff.w < 0.0 ? vec4(-diff.xyz, -diff.w) : vec4(diff.xyz, abs(diff.w));
    precise vec4 t = d * inv_k;
    precise vec4 h = vec4(-t.xyz, 1.0 - t.w);
    h = 0.0 > h.w ? vec4(0.0) : h;
    precise vec4 m = b.w < a.w ? b : a;
    precise vec4 h2 = vec4(h.xyz * h.w + h.w * h.xyz, h.w * h.w);
    precise vec4 amt = h2 * (0.25 * k);
    return vec4(m.xyz - amt.xyz, m.w - amt.w);
}

vec4 arch_smooth_max(vec4 a, vec4 b, float k, float inv_k) {
    precise vec4 s = arch_smooth_min(vec4(-a.xyz, -a.w), vec4(-b.xyz, -b.w), k, inv_k);
    return vec4(-s.xyz, -s.w);
}

// record_wins (ops 0 and 1 are the unions: static_assert at record_wins).
bool arch_record_wins(vec4 r, vec4 c, uint op) {
    return op <= 1u ? c.w < r.w : -c.w > r.w;
}

vec4 arch_combine(vec4 r, vec4 c, DefRecord rec) {
    uint op = rec.head.y;
    if (op == 0u) {
        return c.w < r.w ? c : r;
    } else if (op == 1u) {
        return arch_smooth_min(r, c, uintBitsToFloat(rec.head.z), uintBitsToFloat(rec.head.w));
    } else if (op == 2u) {
        precise vec4 n = vec4(-c.xyz, -c.w);
        return n.w > r.w ? n : r;
    }
    precise vec4 n = vec4(-c.xyz, -c.w);
    return arch_smooth_max(r, n, uintBitsToFloat(rec.head.z), uintBitsToFloat(rec.head.w));
}

// moment_surface.
vec4 arch_moments(vec4 r, vec4 phi, int lod, uint best, inout uint material) {
    precise float h = exact_pow2(lod);
    precise vec4 lift = vec4(r.x - phi.x * h, r.y - phi.y * h, r.z - phi.z * h, r.w - phi.w * h);
    precise float n2 = phi.x * phi.x + phi.y * phi.y + phi.z * phi.z;
    precise float gh = det_rsqrt(max(n2, MOMENT_GRAD_FLOOR)) * h;
    precise vec4 occ = vec4(-phi.x * gh, -phi.y * gh, -phi.z * gh, max((MOMENT_ISO - phi.w) * gh, -(MOMENT_DEPTH_CELLS * h)));
    if (best != INHERIT_INDEXu && phi.w >= MOMENT_MATERIAL_ISO) {
        material = best;
    }
    return occ.w < lift.w ? occ : lift;
}
)glsl";

ov::str generate_library_glsl(const DefLibrary &p_library) {
    ov::str code = "// Archetype library: one function per type, reading the instance's DefRecord.\n";
    for (int i = 0; i < p_library.size(); i++) {
        code += generate_def_glsl(p_library[i].program, "arch_" + ov::itos(i), false);
        code += "\n";
    }
    return code;
}

// arch_fold: fold_onto per sample (p_group_size 0), or per workgroup of p_group_size samples, its list tested once
// against the group's box; the same loop body text either way.
static ov::str fold_loop_glsl(uint32_t p_group_size) {
    // fold_onto's loop body for list word `word` (the same text in both loops: `continue` moves to the next word).
    const char *const FOLD_BODY = R"glsl(        DefRecord rec = sdf_records[word & LIST_ID_MASKu];
        if (!arch_in_box(p, rec, lod)) {
            continue;
        }
        precise vec4 c = arch_eval_instance(p, lod, rec);
        uint index = word >> LIST_ID_BITSu;
        if (rec.head.y == ACCUMu) {
            phi = vec4(phi.x + c.x, phi.y + c.y, phi.z + c.z, phi.w + c.w);
            if (c.w > best) {
                best = c.w;
                best_index = index;
            }
            blobs = true;
            touched = true;
            continue;
        }
        c.w = c.w + intBitsToFloat(rec.box_hi.w) * max(exact_pow2(lod) - intBitsToFloat(rec.anchor_hi.w), 0.0);
        if (index != INHERIT_INDEXu && arch_record_wins(r, c, rec.head.y)) {
            material = index;
        }
        r = arch_combine(r, c, rec);
        touched = true;
)glsl";
    const char *const FOLD_LOCALS = R"glsl(uint off = g_sdf_chunk.id.w;
    uint n = g_sdf_chunk.anchor_hi.w;
    precise vec4 phi = vec4(0.0);
    bool blobs = false;
    float best = 0.0;
    uint best_index = INHERIT_INDEXu;
)glsl";
    ov::str loop;
    if (p_group_size == 0) {
        loop = ov::str(R"glsl(vec4 arch_fold(vec4 r, WorldPos p, int lod, inout bool touched, inout uint material, inout bool moments) {
    FOLD_LOCALS    for (uint i = 0u; i < n; i++) {
        uint word = sdf_list[off + i];
FOLD_BODY    }
)glsl");
    } else {
        OV_ERR_FAIL_COND_V_MSG(p_group_size % 32 != 0, ov::str(), "generate_fold_glsl: the workgroup size must be a multiple of 32.");
        // A slice of GROUP words: each invocation tests one against the group's box, the passes are a bit mask (word
        // h of the mask, bit b: list index base + 32 h + b), walked lowest bit first, so in list order. n is the
        // chunk's: the loop and its barriers are uniform.
        loop = ov::replace_all(ov::str(R"glsl(shared uint s_fold_word[GROUP];
shared uint s_fold_mask[MASK_WORDS];

vec4 arch_fold(vec4 r, WorldPos p, WorldPos group_lo, WorldPos group_hi, int lod, inout bool touched, inout uint material, inout bool moments) {
    FOLD_LOCALS    uint t = gl_LocalInvocationIndex;
    for (uint base = 0u; base < n; base += GROUPu) {
        if (t < MASK_WORDSu) {
            s_fold_mask[t] = 0u;
        }
        barrier();
        if (base + t < n) {
            uint word = sdf_list[off + base + t];
            s_fold_word[t] = word;
            if (arch_meets_box(group_lo, group_hi, sdf_records[word & LIST_ID_MASKu], lod)) {
                atomicOr(s_fold_mask[t >> 5u], 1u << (t & 31u));
            }
        }
        barrier();
        for (uint h = 0u; h < MASK_WORDSu; h++) {
            uint bits = s_fold_mask[h];
            while (bits != 0u) {
                uint word = s_fold_word[h * 32u + uint(findLSB(bits))];
                bits &= bits - 1u;
FOLD_BODY            }
        }
        barrier(); // the next slice overwrites s_fold_word / s_fold_mask
    }
)glsl"),
                { { "MASK_WORDS", ov::itos(p_group_size / 32) }, { "GROUP", ov::itos(p_group_size) } });
    }
    return (ov::replace_all(loop, { { "FOLD_LOCALS", FOLD_LOCALS }, { "FOLD_BODY", FOLD_BODY } }) + R"glsl(    moments = moments || blobs;
    return blobs ? arch_moments(r, phi, lod, best_index, material) : r;
}

)glsl");
}

ov::str generate_fold_glsl(const DefLibrary &p_library, float p_out_scale, bool p_normalize, uint32_t p_group_size) {
    ov::str code = ov::replace_all(ov::str(GLSL_TABLE_FOLD), { { "LIMIT", ov::itos(INPUT_LIMIT) }, { "MOMENT_DEPTH_CELLS", glsl_float(MOMENT_DEPTH_CELLS) }, { "MOMENT_GRAD_FLOOR", glsl_float(MOMENT_GRAD_FLOOR) }, { "MOMENT_MATERIAL_ISO", glsl_float(MOMENT_MATERIAL_ISO) }, { "MOMENT_ISO", glsl_float(MOMENT_ISO) }, { "INHERIT_INDEX", ov::itos(ChunkPalette::INHERIT_INDEX) } });
    code += "\n";
    code += generate_library_glsl(p_library);
    code += "vec4 arch_eval_instance(WorldPos p, int lod, DefRecord rec) {\n    switch (rec.head.x) {\n";
    for (int i = 0; i < p_library.size(); i++) {
        code += ov::format("        case %du: return arch_%d(p, lod, rec);\n", i, i);
    }
    code += "        default: return vec4(0.0);\n    }\n}\n\n";
    code += ov::replace_all(fold_loop_glsl(p_group_size), { { "LIST_ID_MASK", ov::itos(SDFGpuTable::LIST_ID_MASK) }, { "LIST_ID_BITS", ov::itos(SDFGpuTable::LIST_ID_BITS) }, { "INHERIT_INDEX", ov::itos(ChunkPalette::INHERIT_INDEX) }, { "ACCUM", ov::itos(int(Combine::ACCUM)) } });
    // finish_table.
    code += "vec4 arch_finish(vec4 o, int lod) {\n";
    if (p_normalize) {
        code += "    precise float s = exact_pow2(lod);\n";
    }
    code += "    precise float v = o.w * " + glsl_float(p_out_scale) + ";\n";
    if (p_normalize) {
        code += "    precise float gr = det_rsqrt(max(o.x * o.x + o.y * o.y + o.z * o.z, " + glsl_float(exact_pow2(GRAD_FLOOR_EXP)) +
                " * s * s));\n";
        code += "    return vec4(o.xyz * gr, v);\n}\n\n";
    } else {
        code += "    return vec4(o.xyz, v);\n}\n\n";
    }
    return code;
}

// ---- library types ----

namespace {

static std::shared_ptr<TableDef> make_def(const char *p_name, int p_param_count, Builder &p_b, Reg p_out) {
    std::shared_ptr<TableDef> d = std::make_shared<TableDef>();
    d->name = p_name;
    d->param_count = p_param_count;
    d->program = p_b.finish(p_out);
    return d;
}

// The bounds of a cube of half size p[0] (a sphere of radius p[0]; the die, the tunnelled cube: their CSG stays
// inside it), L = 1.
void cube_bounds(const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
    r_lo = Float3{ -p[0], -p[0], -p[0] };
    r_hi = Float3{ p[0], p[0], p[0] };
    r_l = 1.0f;
}

// The bounds of a box of half sizes p[0], p[1], p[2] (anything carved out of it), L = 1.
void box_bounds(const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
    r_lo = Float3{ -p[0], -p[1], -p[2] };
    r_hi = Float3{ p[0], p[1], p[2] };
    r_l = 1.0f;
}

} // namespace

TableDefPtr make_sphere_type() {
    Builder b;
    const Reg r = b.sphere(b.t_local(), b.t_param(0));
    std::shared_ptr<TableDef> d = make_def("sphere", 1, b, r);
    d->bounds = cube_bounds;
    d->moments = [](const float *p) { return sphere_moments(p[0]); };
    return d;
}

TableDefPtr make_box_type() {
    Builder b;
    const Reg r = b.box(b.t_local(), b.t_param(0), b.t_param(1), b.t_param(2));
    std::shared_ptr<TableDef> d = make_def("box", 3, b, r);
    d->bounds = box_bounds;
    d->moments = [](const float *p) { return box_moments(p[0], p[1], p[2]); };
    return d;
}

TableDefPtr make_capsule_type() {
    Builder b;
    const Reg r = b.capsule(b.t_local(), b.t_param(0), b.t_param(1));
    std::shared_ptr<TableDef> d = make_def("capsule", 2, b, r);
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        r_lo = Float3{ -p[1], -p[1], -p[1] };
        r_hi = Float3{ p[1], p[0] + p[1], p[1] };
        r_l = 1.0f;
    };
    d->moments = [](const float *p) { return tapered_capsule_moments(p[0], p[1], p[1]); };
    return d;
}

TableDefPtr make_branch_type() {
    // params {height, radius, slope}: a capsule from y = 0 to y = height whose radius goes linearly from radius
    // to radius + slope * height (slope = (r1 - r0) / height, computed on the CPU: no division here). Not an
    // exact distance (the radius changes along the axis, not normal to the surface), but outside its box it is
    // at least the distance to the capsule of the larger radius. Its gradient is the unit radial direction minus
    // slope along the axis (normal to it), so |grad| <= sqrt(1 + slope^2) <= 1 + slope^2 / 2.
    Builder b;
    const Local l = b.t_local();
    const Reg yc = b.clamp(l.y, b.constant(0.0f), b.t_param(0));
    const Reg len = b.length(l.x, b.sub(l.y, yc), l.z);
    const Reg r = b.sub(len, b.add(b.t_param(1), b.mul(b.t_param(2), yc)));
    std::shared_ptr<TableDef> d = make_def("branch", 3, b, r);
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float rad = std::max(p[1], p[1] + p[2] * p[0]);
        r_lo = Float3{ -rad, -rad, -rad };
        r_hi = Float3{ rad, p[0] + rad, rad };
        r_l = 1.0f + 0.5f * p[2] * p[2];
    };
    d->moments = [](const float *p) { return tapered_capsule_moments(p[0], p[1], double(p[1]) + double(p[2]) * double(p[0])); };
    return d;
}

TableDefPtr make_cylinder_type() {
    Builder b;
    const Reg r = b.cylinder(b.t_local(), b.t_param(0), b.t_param(1));
    std::shared_ptr<TableDef> d = make_def("cylinder", 2, b, r);
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        r_lo = Float3{ -p[1], -p[0], -p[1] };
        r_hi = Float3{ p[1], p[0], p[1] };
        r_l = 1.0f;
    };
    d->moments = [](const float *p) { return cylinder_moments(p[0], p[1]); };
    return d;
}

namespace {

// The rock's and the cone's bumps: fbm_unit, octaves k .. k + 2 at amplitude 2^-i (the sum is within ~2.5), times
// the amp param p_amp. r_reach / r_slope: how far it moves the surface and how much it adds to L, per unit of amp.
Reg bumps(Builder &p_b, int p_k, uint32_t p_seed, Reg p_amp, double &r_reach, double &r_slope) {
    Reg noise = p_b.constant(0.0f);
    r_reach = 0.0;
    r_slope = 0.0;
    for (int i = 0; i < 3; i++) {
        const float amp = float(std::ldexp(1.0, -i));
        noise = p_b.add(noise, p_b.t_perlin3(p_k + i, p_seed + uint32_t(i), Vector3i64(), amp));
        r_reach += double(amp) * PERLIN3_MAX;
        r_slope += double(amp) * PERLIN3_GRAD_MAX * std::ldexp(1.0, -(p_k + i));
    }
    return p_b.mul(noise, p_amp);
}

} // namespace

TableDefPtr make_rock_type(int p_k, uint32_t p_seed) {
    // params {radius, amp}: sphere(radius) + bumps. The lattice is fixed by p_k, so the bumps keep one size; the
    // demo picks p_k from the nominal radius.
    Builder b;
    const Local l = b.t_local();
    const Reg s = b.sphere(l, b.t_param(0));
    double reach = 0.0, slope = 0.0;
    const Reg r = b.add(s, bumps(b, p_k, p_seed, b.t_param(1), reach, slope));
    std::shared_ptr<TableDef> d = make_def("rock", 2, b, r);
    d->bounds = [reach, slope](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float e = float(p[0] + double(p[1]) * reach);
        r_lo = Float3{ -e, -e, -e };
        r_hi = Float3{ e, e, e };
        r_l = float(1.0 + double(p[1]) * slope);
    };
    d->moments = [](const float *p) { return sphere_moments(p[0]); };
    return d;
}

TableDefPtr make_cone_type(int p_k, uint32_t p_seed) {
    // params {radius, height, cos, sin, amp}: max(|xz| cos + y sin - radius cos, -y, y - height) + bumps. The first
    // term is the distance to the lateral surface (the cone from radius at y = 0 to its apex at y = height, cos and
    // sin of its half angle), 1-Lipschitz like the caps; the top cap only bounds the bumps (without it they would
    // raise the apex by reach / sin). The rim (lateral and base) is a smooth max over half a bump wavelength: a sharp
    // crease under the bumps meshes ragged. A smooth max is >= the max (the bounds hold) and still 1-Lipschitz.
    Builder b;
    const Local l = b.t_local();
    const Reg lateral = b.sub(b.add(b.mul(b.length(l.x, l.z), b.t_param(2)), b.mul(l.y, b.t_param(3))), b.mul(b.t_param(0), b.t_param(2)));
    const Reg caps = b.max(b.neg(l.y), b.sub(l.y, b.t_param(1)));
    double reach = 0.0, slope = 0.0;
    const Reg r = b.add(b.smooth_max(lateral, caps, float(std::ldexp(1.0, p_k - 1))), bumps(b, p_k, p_seed, b.t_param(4), reach, slope));
    std::shared_ptr<TableDef> d = make_def("cone", 5, b, r);
    d->bounds = [reach, slope](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float bump = float(double(p[4]) * reach);
        // lateral <= bump with y >= -bump: |xz| <= radius + bump (1 + sin) / cos.
        const float e = float(double(p[0]) + double(bump) * (1.0 + double(p[3])) / double(p[2]));
        r_lo = Float3{ -e, -bump, -e };
        r_hi = Float3{ e, p[1] + bump, e };
        r_l = float(1.0 + double(p[4]) * slope);
    };
    d->moments = [](const float *p) { // the radius goes from r at y = 0 to 0 at the apex: (r - r y / h)^2
        const double r = p[0], h = p[1];
        return revolution_moments(r * r, -2.0 * r * r / h, r * r / (h * h), 0.0, h);
    };
    return d;
}

TableDefPtr make_basin_type(int p_k, uint32_t p_seed) {
    // params {half_height, radius, amp}: a cylinder + bumps, a lava lake's crater (biome_features.hpp).
    Builder b;
    double reach = 0.0, slope = 0.0;
    const Reg r = b.add(b.cylinder(b.t_local(), b.t_param(0), b.t_param(1)), bumps(b, p_k, p_seed, b.t_param(2), reach, slope));
    std::shared_ptr<TableDef> d = make_def("basin", 3, b, r);
    d->bounds = [reach, slope](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float bump = float(double(p[2]) * reach);
        r_lo = Float3{ -(p[1] + bump), -(p[0] + bump), -(p[1] + bump) };
        r_hi = Float3{ p[1] + bump, p[0] + bump, p[1] + bump };
        r_l = float(1.0 + double(p[2]) * slope);
    };
    d->moments = [](const float *p) { return cylinder_moments(p[0], p[1]); };
    return d;
}

TableDefPtr make_island_type(int p_k, uint32_t p_seed) {
    // params {radius, height, cos, sin, dome_radius, dome_drop, amp}: a floating island, its rim the circle of radius
    // `radius` on y = 0. The top is a spherical cap: the sphere of dome_radius centred dome_drop below the rim, cut at
    // y = 0 (dome_radius^2 - dome_drop^2 = radius^2, from the CPU). The underside is the cone hanging from the rim to
    // its apex at y = -height (make_cone_type's lateral term upside down, cos / sin of its half angle). Both are smooth-
    // unioned over half a bump wavelength, so the rim is no crease, then the bumps are added. The smooth min is at most
    // blend / 4 below the min: the bounds grow by that.
    Builder b;
    const Local l = b.t_local();
    const Reg up = b.neg(l.y);
    const Reg dome = b.max(b.sphere(Local{ l.x, b.add(l.y, b.t_param(5)), l.z }, b.t_param(4)), up);
    const Reg lateral = b.sub(b.add(b.mul(b.length(l.x, l.z), b.t_param(2)), b.mul(up, b.t_param(3))), b.mul(b.t_param(0), b.t_param(2)));
    const Reg cone = b.max(lateral, b.max(l.y, b.sub(up, b.t_param(1))));
    const float blend = float(std::ldexp(1.0, p_k - 1));
    double reach = 0.0, slope = 0.0;
    const Reg r = b.add(b.smooth_min(dome, cone, blend), bumps(b, p_k, p_seed, b.t_param(6), reach, slope));
    std::shared_ptr<TableDef> d = make_def("island", 7, b, r);
    d->bounds = [reach, slope, blend](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const double grow = double(p[6]) * reach + 0.25 * double(blend);
        // The cone: lateral <= grow with y <= grow gives |xz| <= radius + grow (1 + sin) / cos. The cap: sphere <= grow
        // with y >= -grow.
        const double cone_e = double(p[0]) + grow * (1.0 + double(p[3])) / double(p[2]);
        const double rs = double(p[4]) + grow, low = double(p[5]) - grow;
        const double dome_e = low > 0.0 ? std::sqrt(ov::max(rs * rs - low * low, 0.0)) : rs;
        const float e = float(ov::max(cone_e, dome_e));
        r_lo = Float3{ -e, float(-double(p[1]) - grow), -e };
        r_hi = Float3{ e, float(double(p[4]) - double(p[5]) + grow), e };
        r_l = float(1.0 + double(p[6]) * slope);
    };
    d->moments = [](const float *p) { // the hanging cone (r + y tan)^2, then the cap rd^2 - (y + drop)^2
        const double r = p[0], tan = double(p[3]) / double(p[2]), rd = p[4], drop = p[5];
        // Down to its apex (y = -r / tan) or its bottom cap, whichever comes first.
        Moments m = revolution_moments(r * r, 2.0 * r * tan, tan * tan, ov::max(-double(p[1]), -r / tan), 0.0);
        m.add(revolution_moments(rd * rd - drop * drop, -2.0 * drop, -1.0, 0.0, rd - drop));
        return m;
    };
    return d;
}

TableDefPtr make_swiss_cube_type(int p_holes) {
    // params {hx, hy, hz, ring_radius, hole_radius}: a box with p_holes spherical holes around its vertical
    // axis. A composite: its holes are children in the top frame, so it turns as a whole.
    Builder b;
    const Local l = b.t_local();
    Reg r = b.box(l, b.t_param(0), b.t_param(1), b.t_param(2));
    for (int i = 0; i < p_holes; i++) {
        const double a = 2.0 * M_PI * double(i) / double(p_holes);
        const Local q{ b.sub(l.x, b.mul(b.t_param(3), float(std::cos(a)))), l.y,
            b.sub(l.z, b.mul(b.t_param(3), float(std::sin(a)))) };
        const Reg hole = b.sphere(q, b.t_param(4));
        r = b.max(r, b.neg(hole));
    }
    std::shared_ptr<TableDef> d = make_def("swiss_cube", 5, b, r);
    d->bounds = box_bounds;
    return d;
}

namespace {

// p_l moved by -p_h (p_x, p_y, p_z): a point of the instance's frame in units of the half size p_h.
Local shifted(Builder &p_b, const Local &p_l, Reg p_h, float p_x, float p_y, float p_z) {
    return Local{ p_b.sub(p_l.x, p_b.mul(p_h, p_x)), p_b.sub(p_l.y, p_b.mul(p_h, p_y)), p_b.sub(p_l.z, p_b.mul(p_h, p_z)) };
}

} // namespace

TableDefPtr make_die_type() {
    // A cube intersected with a sphere of 1.55 h, pips of 0.2 h on a 3 x 3 grid of h / 2 steps subtracted. Hard CSG
    // (max of 1-Lipschitz bounds): sharp creases for dual contouring.
    Builder b;
    const Local l = b.t_local();
    const Reg h = b.t_param(0);
    Reg r = b.max(b.box(l, h, h, h), b.sphere(l, b.mul(h, 1.55f)));
    const char *faces[5] = { "000010000", "100000001", "100010001", "101000101", "101010101" };
    const int axis[5] = { 1, 2, 0, 0, 2 };
    const float sign[5] = { 1.0f, 1.0f, 1.0f, -1.0f, -1.0f };
    const Reg pip = b.mul(h, 0.2f);
    for (int f = 0; f < 5; f++) {
        for (int g = 0; g < 9; g++) {
            if (faces[f][g] == '1') {
                float c[3];
                c[axis[f]] = sign[f];
                c[(axis[f] + 1) % 3] = float(g % 3 - 1) * 0.5f;
                c[(axis[f] + 2) % 3] = float(g / 3 - 1) * 0.5f;
                r = b.max(r, b.neg(b.sphere(shifted(b, l, h, c[0], c[1], c[2]), pip)));
            }
        }
    }
    std::shared_ptr<TableDef> d = make_def("die", 1, b, r);
    d->bounds = cube_bounds;
    return d;
}

TableDefPtr make_stairs_type() {
    Builder b;
    const Local l = b.t_local();
    Reg r = -1;
    for (int i = 0; i < 8; i++) {
        const Local q{ l.x, l.y, b.sub(l.z, b.mul(b.t_param(2), float(2 * i))) };
        const Reg step = b.box(q, b.t_param(0), b.mul(b.t_param(1), float(i + 1)), b.t_param(2));
        r = r < 0 ? step : b.min(r, step);
    }
    std::shared_ptr<TableDef> d = make_def("stairs", 3, b, r);
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        r_lo = Float3{ -p[0], -8.0f * p[1], -p[2] };
        r_hi = Float3{ p[0], 8.0f * p[1], 15.0f * p[2] };
        r_l = 1.0f;
    };
    d->moments = [](const float *p) { // eight boxes side by side along z
        Moments m;
        for (int i = 0; i < 8; i++) {
            const double at[3] = { 0.0, 0.0, 2.0 * i * double(p[2]) };
            m.add(box_moments(p[0], double(p[1]) * (i + 1), p[2]), at);
        }
        return m;
    };
    return d;
}

TableDefPtr make_tunnel_cube_type() {
    // The tunnels run 1.25 h each way: through the faces.
    Builder b;
    const Local l = b.t_local();
    const Reg h = b.t_param(0);
    const Reg t = b.t_param(1);
    const Reg through = b.mul(h, 1.25f);
    Reg r = b.box(l, h, h, h);
    r = b.max(r, b.neg(b.box(l, through, t, t)));
    r = b.max(r, b.neg(b.box(l, t, through, t)));
    r = b.max(r, b.neg(b.box(l, t, t, through)));
    std::shared_ptr<TableDef> d = make_def("tunnel_cube", 2, b, r);
    d->bounds = cube_bounds;
    return d;
}

TableDefPtr make_moment_blob_type() {
    Builder b;
    const Local l = b.t_local();
    const Reg q = b.add(b.add(b.mul(b.mul(l.x, l.x), b.t_param(0)), b.mul(b.mul(l.y, l.y), b.t_param(1))), b.mul(b.mul(l.z, l.z), b.t_param(2)));
    const Reg t = b.max(b.sub(b.constant(1.0f), q), b.constant(0.0f));
    const Reg r = b.mul(b.mul(t, t), b.t_param(3));
    std::shared_ptr<TableDef> d = make_def("moment_blob", 4, b, r);
    // Zero outside the ellipsoid of semi-axes 1 / sqrt(p_i): its box. Not a distance (L = 0: the locality margin is
    // the skirt's alone; outside its box it adds exactly 0 to phi).
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float s[3] = { float(1.0 / std::sqrt(double(p[0]))), float(1.0 / std::sqrt(double(p[1]))), float(1.0 / std::sqrt(double(p[2]))) };
        r_lo = Float3{ -s[0], -s[1], -s[2] };
        r_hi = Float3{ s[0], s[1], s[2] };
        r_l = 0.0f;
    };
    return d;
}

namespace {

// Corner p_c (bits x, y, z: the high side) of the box (p_lo, p_hi).
void box_corner(Float3 p_lo, Float3 p_hi, int p_c, double r_q[3]) {
    r_q[0] = (p_c & 1) ? p_hi.x : p_lo.x;
    r_q[1] = (p_c & 2) ? p_hi.y : p_lo.y;
    r_q[2] = (p_c & 4) ? p_hi.z : p_lo.z;
}

} // namespace

double rotation_margin(Float3 p_lo, Float3 p_hi) {
    double reach = 0.0;
    for (int c = 0; c < 8; c++) {
        double q[3];
        box_corner(p_lo, p_hi, c, q);
        reach = ov::max(reach, std::abs(q[0]) + std::abs(q[1]) + std::abs(q[2]));
    }
    return 1.0 + 1e-4 * reach;
}

Bounds world_bounds(Float3 p_lo, Float3 p_hi, float p_lipschitz, const VoxelPos &p_anchor, const float *p_r) {
    Bounds b;
    b.lipschitz = p_lipschitz;
    double lo[3] = { INFINITY, INFINITY, INFINITY };
    double hi[3] = { -INFINITY, -INFINITY, -INFINITY };
    for (int c = 0; c < 8; c++) {
        double q[3];
        box_corner(p_lo, p_hi, c, q);
        double w[3];
        local_to_world(p_r, q, w);
        for (int j = 0; j < 3; j++) {
            lo[j] = ov::min(lo[j], w[j]);
            hi[j] = ov::max(hi[j], w[j]);
        }
    }
    const double margin = rotation_margin(p_lo, p_hi);
    for (int j = 0; j < 3; j++) {
        b.lo[j] = p_anchor[j] + int64_t(std::floor(lo[j] - margin));
        b.hi[j] = p_anchor[j] + int64_t(std::ceil(hi[j] + margin));
    }
    return b;
}

Bounds record_world_bounds(const TableDef &p_def, const TableRecord &p_record) {
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    return world_bounds(lo, hi, lipschitz, p_record.anchor, p_record.rotation);
}

ov::str record_fit_error(const TableDef &p_def, const TableRecord &p_record, int64_t p_max_extent) {
    for (int a = 0; a < 3; a++) {
        if (p_record.anchor[a] > WorldCoord::MAX_ABS_ORIGIN_VOXEL || p_record.anchor[a] < -WorldCoord::MAX_ABS_ORIGIN_VOXEL) {
            return "the anchor is beyond +-2^60 voxels.";
        }
    }
    const double smooth = 2.0 * double(p_record.k);
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    // The influence margin grows by 2 L voxels per voxel of the LOD (finish_record's margin_c): a steep field
    // would grow the box like a huge one.
    if (!(lipschitz >= 0.0f && lipschitz <= MAX_RECORD_LIPSCHITZ)) {
        return ov::format("the type's Lipschitz bound %f for these params is outside [0, %f].", lipschitz, MAX_RECORD_LIPSCHITZ);
    }
    for (const float *v : { &lo.x, &lo.y, &lo.z, &hi.x, &hi.y, &hi.z }) {
        if (!std::isfinite(*v)) {
            return "the type's box is not finite for these params.";
        }
    }
    // Any rotation keeps the box inside the sphere through its farthest corner.
    double reach2 = 0.0;
    const float far[3] = { std::max(std::fabs(lo.x), std::fabs(hi.x)), std::max(std::fabs(lo.y), std::fabs(hi.y)), std::max(std::fabs(lo.z), std::fabs(hi.z)) };
    for (float f : far) {
        reach2 += double(f) * double(f);
    }
    const double extent = 2.0 * (std::sqrt(reach2) + rotation_margin(lo, hi)) + smooth;
    if (!(extent <= double(p_max_extent))) {
        return ov::format("the instance spans %.0f voxels (its box plus twice k), more than the %d the table takes.", extent, p_max_extent);
    }
    return ov::str();
}

void finish_table_record(const DefLibrary &p_library, const RecordFinish &p_finish, TableRecord &r_record) {
    r_record.material = record_material(r_record.material, p_library.material(r_record.def), r_record.op);
    finish_record(p_library[r_record.def], r_record, p_finish.skirt_depth, p_finish.band, p_finish.rule);
}

} // namespace Archetype
