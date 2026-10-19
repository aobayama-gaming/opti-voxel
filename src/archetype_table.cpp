// The archetype table: the library, the records, the locality rule and the fold (archetype_table.hpp).
#include "archetype_table.hpp"

#include <cmath>
#include <cstring>

namespace Archetype {

using ExactNoise::exact_pow2;
using ExactNoise::glsl_float;

DefRecord TableRecord::def_record() const {
    DefRecord r;
    r.anchor = anchor;
    std::memcpy(r.rotation, rotation, sizeof(r.rotation));
    std::memcpy(r.params, params, sizeof(r.params));
    r.salt = salt;
    return r;
}

RecordStore::RecordStore(const std::vector<TableRecord> &p_records, uint32_t p_version) {
    for (const TableRecord &r : p_records) {
        push_back(r, p_version);
    }
}

RecordStore::Page &RecordStore::own(uint32_t p_page) {
    if (pages[p_page].use_count() > 1) {
        pages[p_page] = std::make_shared<Page>(*pages[p_page]); // a snapshot shares it: copy on write
    }
    return *pages[p_page];
}

void RecordStore::push_back(const TableRecord &p_record, uint32_t p_version) {
    if (count % PAGE == 0) {
        pages.push_back(std::make_shared<Page>());
        pages.back()->r.reserve(PAGE);
    }
    Page &page = own(uint32_t(pages.size() - 1));
    page.r.push_back(p_record);
    page.version = MAX(page.version, p_version);
    count++;
}

void RecordStore::set(uint32_t p_i, const TableRecord &p_record, uint32_t p_version) {
    Page &page = own(p_i / PAGE);
    page.r[p_i % PAGE] = p_record;
    page.version = MAX(page.version, p_version);
}

void RecordStore::changed_since(uint32_t p_version, std::vector<std::pair<uint32_t, uint32_t>> &r_runs) const {
    r_runs.clear();
    for (uint32_t p = 0; p < pages.size(); p++) {
        if (p_version != UINT32_MAX && pages[p]->version <= p_version) {
            continue;
        }
        const uint32_t first = p * PAGE;
        const uint32_t end = MIN(count, (p + 1) * PAGE);
        if (!r_runs.empty() && r_runs.back().first + r_runs.back().second == first) {
            r_runs.back().second = end - r_runs.back().first;
        } else {
            r_runs.push_back({ first, end - first });
        }
    }
}

int DefLibrary::add(const TableDefPtr &p_def) {
    list.push_back(p_def);
    return int(list.size() - 1);
}

int influence_margin(const TableRecord &p_record, int p_lod) {
    // Clamped before the conversion: a float above INT_MAX converts to anything (CPU) or saturates (GPU).
    const float m = MIN(p_record.k + p_record.margin_c * exact_pow2(p_lod), float(INPUT_LIMIT));
    return int(std::ceil(m)) + 2;
}

double band_depth(const TableDef &p_def, const TableRecord &p_record) {
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    double side = MIN(MIN(double(hi.x) - double(lo.x), double(hi.y) - double(lo.y)), double(hi.z) - double(lo.z)) + 2.0 * rotation_margin(lo, hi);
    for (int a = 0; a < 3; a++) {
        side = MIN(side, double(p_record.box_hi[a]) - double(p_record.box_lo[a]));
    }
    return 0.5 * side * double(p_record.lipschitz) + double(p_record.k);
}

void set_band(TableRecord &r_record, const TableDef &p_def, bool p_enabled, float p_erode_cells) {
    r_record.max_lod = TableRecord::NEVER_DROPPED;
    r_record.erode = 0.0f;
    if (!p_def.bounded || !p_enabled) {
        return;
    }
    r_record.erode = p_erode_cells;
    const double depth = band_depth(p_def, r_record);
    for (int lod = 0; lod < 32; lod++) {
        if (double(p_erode_cells) * (std::ldexp(1.0, lod) - 1.0) >= depth) {
            r_record.max_lod = uint8_t(lod);
            break;
        }
    }
}

void finish_record(const TableDef &p_def, TableRecord &r_record, float p_skirt_depth, bool p_band, float p_erode_cells) {
    const Bounds wb = record_world_bounds(p_def, r_record);
    for (int a = 0; a < 3; a++) {
        r_record.box_lo[a] = int32_t(wb.lo[a] - r_record.anchor[a]);
        r_record.box_hi[a] = int32_t(wb.hi[a] - r_record.anchor[a]);
    }
    r_record.lipschitz = wb.lipschitz;
    r_record.margin_c = 2.0f * wb.lipschitz + p_skirt_depth;
    set_band(r_record, p_def, p_band, p_erode_cells);
}

bool in_influence_box(const TableRecord &p_record, const WorldPos &p_pos, int p_lod) {
    const int m = influence_margin(p_record, p_lod);
    for (int a = 0; a < 3; a++) {
        const int64_t d = WorldCoord::sat_diff(p_pos.voxel[a], p_record.anchor[a], INPUT_LIMIT);
        if (d < int64_t(p_record.box_lo[a]) - m || d > int64_t(p_record.box_hi[a]) + m) {
            return false;
        }
    }
    return true;
}

void encode_record(PackedByteArray &r_bytes, uint32_t p_index, const TableRecord &p_record) {
    using namespace WorldCoord;
    const int64_t base = int64_t(p_index) * RECORD_BYTES;
    r_bytes.encode_u32(base + 0, p_record.def);
    r_bytes.encode_u32(base + 4, uint32_t(p_record.op));
    r_bytes.encode_float(base + 8, p_record.k);
    r_bytes.encode_float(base + 12, p_record.inv_k);
    r_bytes.encode_u32(base + 16, lo32(p_record.anchor.x));
    r_bytes.encode_u32(base + 20, lo32(p_record.anchor.y));
    r_bytes.encode_u32(base + 24, lo32(p_record.anchor.z));
    r_bytes.encode_u32(base + 28, p_record.salt);
    r_bytes.encode_s32(base + 32, hi32(p_record.anchor.x));
    r_bytes.encode_s32(base + 36, hi32(p_record.anchor.y));
    r_bytes.encode_s32(base + 40, hi32(p_record.anchor.z));
    r_bytes.encode_s32(base + 44, 0);
    for (int i = 0; i < 9; i++) {
        r_bytes.encode_float(base + 48 + (i / 3) * 16 + (i % 3) * 4, p_record.rotation[i]);
    }
    for (int i = 0; i < 8; i++) {
        r_bytes.encode_float(base + 96 + i * 4, p_record.params[i]);
    }
    for (int i = 0; i < 3; i++) {
        r_bytes.encode_s32(base + 128 + i * 4, p_record.box_lo[i]);
        r_bytes.encode_s32(base + 144 + i * 4, p_record.box_hi[i]);
    }
    r_bytes.encode_float(base + 140, p_record.margin_c);
    r_bytes.encode_float(base + 156, p_record.erode);
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

// The four allowed top-level operators (archetype_instance.hpp Combine), on registers.
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

Float4 fold(const Program &p_ground, const DefLibrary &p_library, RecordView p_records, const uint32_t *p_list,
        uint32_t p_count, const WorldPos &p_pos, int p_lod, bool p_box_test) {
    thread_local std::vector<Float4> regs;
    run_registers(p_ground, p_pos, p_lod, regs);
    Float4 r = regs[p_ground.output];
    for (uint32_t i = 0; i < p_count; i++) {
        const TableRecord &rec = p_records[p_list[i]];
        if (p_box_test && !in_influence_box(rec, p_pos, p_lod)) {
            continue;
        }
        const DefRecord dr = rec.def_record();
        run_registers(p_library[rec.def].program, p_pos, p_lod, regs, &dr);
        Float4 c = regs[p_library[rec.def].program.output];
        c.w = c.w + rec.erode * (exact_pow2(p_lod) - 1.0f); // the band rule's erosion (set_band)
        r = combine4(r, c, rec);
    }
    return r;
}

Float4 finish_table(const Float4 &p_out, int p_lod, float p_out_scale, bool p_normalize) {
    const float v = p_out.w * p_out_scale;
    if (!p_normalize) {
        return Float4{ p_out.x, p_out.y, p_out.z, v };
    }
    const float n2 = p_out.x * p_out.x + p_out.y * p_out.y + p_out.z * p_out.z;
    const float s = exact_pow2(p_lod);
    const float gr = ExactNoise::det_rsqrt(MAX(n2, exact_pow2(GRAD_FLOOR_EXP) * s * s));
    return Float4{ p_out.x * gr, p_out.y * gr, p_out.z * gr, v };
}

// ---- GLSL ----

const char GLSL_DEF_RECORD_STRUCT[] = R"glsl(
struct DefRecord {
    uvec4 head;      // x: def, y: op, z: k bits, w: inv_k bits
    uvec4 anchor_lo; // xyz: low words, w: noise salt
    ivec4 anchor_hi; // xyz: high words, w: unused
    vec4 rot[3];     // world -> local, row major
    float params[8];
    ivec4 box_lo;    // xyz: influence box low (relative), w: margin_c bits
    ivec4 box_hi;    // xyz: influence box high, w: erode bits (voxels per 2^lod - 1, set_band)
};
)glsl";

// arch_in_box / arch_smooth_min / arch_smooth_max / arch_combine: the GPU mirror of influence_margin,
// in_influence_box and combine4. Spliced with the terrain (get_glsl_code), after the noise.
static const char GLSL_TABLE_FOLD[] = R"glsl(

bool arch_in_box(WorldPos p, DefRecord rec, int lod) {
    ivec3 d = sat_diff(p, rec.anchor_lo.xyz, rec.anchor_hi.xyz, LIMIT);
    precise float mf = min(uintBitsToFloat(rec.head.z) + uintBitsToFloat(rec.box_lo.w) * exact_pow2(lod), float(LIMIT));
    int m = int(ceil(mf)) + 2;
    return all(greaterThanEqual(d, rec.box_lo.xyz - ivec3(m))) && all(lessThanEqual(d, rec.box_hi.xyz + ivec3(m)));
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
)glsl";

String generate_library_glsl(const DefLibrary &p_library) {
    String code = "// Archetype library: one function per type, reading the instance's DefRecord.\n";
    for (int i = 0; i < p_library.size(); i++) {
        code += generate_def_glsl(p_library[i].program, "arch_" + itos(i), false);
        code += "\n";
    }
    return code;
}

static String fold_glsl(const DefLibrary &p_library) {
    String code = String(GLSL_TABLE_FOLD).replace("LIMIT", itos(INPUT_LIMIT));
    code += "\nChunkInput g_sdf_chunk;\n\n";
    code += generate_library_glsl(p_library);
    code += "vec4 arch_eval_instance(WorldPos p, int lod, DefRecord rec) {\n    switch (rec.head.x) {\n";
    for (int i = 0; i < p_library.size(); i++) {
        code += vformat("        case %du: return arch_%d(p, lod, rec);\n", i, i);
    }
    code += "        default: return vec4(0.0);\n    }\n}\n\n";
    code += R"glsl(vec4 terrain_sdf_register(WorldPos p, int lod) {
    vec4 r = arch_ground_register(p, lod);
    uint off = g_sdf_chunk.id.w;
    uint n = g_sdf_chunk.anchor_hi.w;
    for (uint i = 0u; i < n; i++) {
        DefRecord rec = sdf_records[sdf_list[off + i]];
        if (!arch_in_box(p, rec, lod)) {
            continue;
        }
        precise vec4 c = arch_eval_instance(p, lod, rec);
        c.w = c.w + intBitsToFloat(rec.box_hi.w) * (exact_pow2(lod) - 1.0);
        r = arch_combine(r, c, rec);
    }
    return r;
}

)glsl";
    return code;
}

String generate_table_sdf(const Program &p_ground, const DefLibrary &p_library, float p_out_scale, bool p_normalize) {
    String code = generate_glsl(p_ground, "arch_ground_register", false);
    code += "\n";
    code += fold_glsl(p_library);
    code += "vec4 terrain_sdf(WorldPos p, int lod) {\n";
    if (p_normalize) {
        code += "    precise float s = exact_pow2(lod);\n";
    }
    code += "    precise vec4 o = terrain_sdf_register(p, lod);\n";
    code += "    precise float v = o.w * " + glsl_float(p_out_scale) + ";\n";
    if (p_normalize) {
        code += "    precise float gr = det_rsqrt(max(o.x * o.x + o.y * o.y + o.z * o.z, " + glsl_float(exact_pow2(GRAD_FLOOR_EXP)) +
                " * s * s));\n";
        code += "    return vec4(o.xyz * gr, v);\n}\n";
    } else {
        code += "    return vec4(o.xyz, v);\n}\n";
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

} // namespace

TableDefPtr make_sphere_type() {
    Builder b;
    const Reg r = b.sphere(b.t_local(), b.t_param(0));
    std::shared_ptr<TableDef> d = make_def("sphere", 1, b, r);
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float rad = p[0];
        r_lo = Float3{ -rad, -rad, -rad };
        r_hi = Float3{ rad, rad, rad };
        r_l = 1.0f;
    };
    return d;
}

TableDefPtr make_box_type() {
    Builder b;
    const Reg r = b.box(b.t_local(), b.t_param(0), b.t_param(1), b.t_param(2));
    std::shared_ptr<TableDef> d = make_def("box", 3, b, r);
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        r_lo = Float3{ -p[0], -p[1], -p[2] };
        r_hi = Float3{ p[0], p[1], p[2] };
        r_l = 1.0f;
    };
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
        const float rad = MAX(p[1], p[1] + p[2] * p[0]);
        r_lo = Float3{ -rad, -rad, -rad };
        r_hi = Float3{ rad, p[0] + rad, rad };
        r_l = 1.0f + 0.5f * p[2] * p[2];
    };
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
    return d;
}

TableDefPtr make_rock_type(int p_k, uint32_t p_seed) {
    // params {radius, amp}: sphere(radius) + amp * fbm_unit, fbm_unit's octaves at amplitude 2^-i (the sum
    // is within ~2.5). The lattice is fixed by p_k, so the bumps keep one size; the demo picks p_k from the
    // nominal radius.
    Builder b;
    const Local l = b.t_local();
    const Reg s = b.sphere(l, b.t_param(0));
    Reg noise = b.constant(0.0f);
    const int octaves = 3;
    double reach_unit = 0.0;
    double slope_unit = 0.0;
    for (int i = 0; i < octaves; i++) {
        const float amp = float(std::ldexp(1.0, -i));
        noise = b.add(noise, b.t_perlin3(p_k + i, p_seed + uint32_t(i), Vector3i64(), amp));
        reach_unit += double(amp) * PERLIN3_MAX;
        slope_unit += double(amp) * PERLIN3_GRAD_MAX * std::ldexp(1.0, -(p_k + i));
    }
    const Reg r = b.add(s, b.mul(noise, b.t_param(1)));
    std::shared_ptr<TableDef> d = make_def("rock", 2, b, r);
    const double reach = reach_unit;
    const double slope = slope_unit;
    d->bounds = [reach, slope](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        const float e = float(p[0] + double(p[1]) * reach);
        r_lo = Float3{ -e, -e, -e };
        r_hi = Float3{ e, e, e };
        r_l = float(1.0 + double(p[1]) * slope);
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
    d->bounds = [](const float *p, Float3 &r_lo, Float3 &r_hi, float &r_l) {
        r_lo = Float3{ -p[0], -p[1], -p[2] };
        r_hi = Float3{ p[0], p[1], p[2] };
        r_l = 1.0f;
    };
    return d;
}

Bounds record_world_bounds(const TableDef &p_def, const TableRecord &p_record) {
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    return world_bounds(lo, hi, lipschitz, p_def.bounded, p_record.anchor, p_record.rotation);
}

String record_fit_error(const TableDef &p_def, const TableRecord &p_record, int64_t p_max_extent) {
    for (int a = 0; a < 3; a++) {
        if (p_record.anchor[a] > WorldCoord::MAX_ABS_ORIGIN_VOXEL || p_record.anchor[a] < -WorldCoord::MAX_ABS_ORIGIN_VOXEL) {
            return "the anchor is beyond +-2^60 voxels.";
        }
    }
    const double smooth = 2.0 * double(p_record.k);
    if (!p_def.bounded) {
        return smooth <= double(p_max_extent) ? String() : vformat("twice k is more than the %d voxels the table takes.", p_max_extent);
    }
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_def.bounds(p_record.params, lo, hi, lipschitz);
    // The influence margin grows by 2 L voxels per voxel of the LOD (finish_record's margin_c): a steep field
    // would grow the box like a huge one.
    if (!(lipschitz >= 0.0f && lipschitz <= MAX_RECORD_LIPSCHITZ)) {
        return vformat("the type's Lipschitz bound %f for these params is outside [0, %f].", lipschitz, MAX_RECORD_LIPSCHITZ);
    }
    for (const float *v : { &lo.x, &lo.y, &lo.z, &hi.x, &hi.y, &hi.z }) {
        if (!std::isfinite(*v)) {
            return "the type's box is not finite for these params.";
        }
    }
    // Any rotation keeps the box inside the sphere through its farthest corner.
    double reach2 = 0.0;
    const float far[3] = { MAX(std::fabs(lo.x), std::fabs(hi.x)), MAX(std::fabs(lo.y), std::fabs(hi.y)), MAX(std::fabs(lo.z), std::fabs(hi.z)) };
    for (float f : far) {
        reach2 += double(f) * double(f);
    }
    const double extent = 2.0 * (std::sqrt(reach2) + rotation_margin(lo, hi)) + smooth;
    if (!(extent <= double(p_max_extent))) {
        return vformat("the instance spans %.0f voxels (its box plus twice k), more than the %d the table takes.", extent, p_max_extent);
    }
    return String();
}

} // namespace Archetype
