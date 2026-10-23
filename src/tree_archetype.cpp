// Trees as archetype-table instances (tree_archetype.hpp).
#include "tree_archetype.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <string>

namespace Archetype {

namespace {

struct Vec {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec operator+(const Vec &p_a, const Vec &p_b) { return Vec{ p_a.x + p_b.x, p_a.y + p_b.y, p_a.z + p_b.z }; }
Vec operator-(const Vec &p_a, const Vec &p_b) { return Vec{ p_a.x - p_b.x, p_a.y - p_b.y, p_a.z - p_b.z }; }
Vec operator*(const Vec &p_a, double p_s) { return Vec{ p_a.x * p_s, p_a.y * p_s, p_a.z * p_s }; }
double dot(const Vec &p_a, const Vec &p_b) { return p_a.x * p_b.x + p_a.y * p_b.y + p_a.z * p_b.z; }
Vec cross(const Vec &p_a, const Vec &p_b) {
    return Vec{ p_a.y * p_b.z - p_a.z * p_b.y, p_a.z * p_b.x - p_a.x * p_b.z, p_a.x * p_b.y - p_a.y * p_b.x };
}
double length(const Vec &p_a) { return std::sqrt(dot(p_a, p_a)); }
Vec unit(const Vec &p_a) { return p_a * (1.0 / length(p_a)); }
// Whole voxels: joints and fork points sit on them, so a segment's end is exactly the next one's anchor.
Vec snap(const Vec &p_a) { return Vec{ std::round(p_a.x), std::round(p_a.y), std::round(p_a.z) }; }

// Two unit vectors normal to the unit p_d, with (e0, p_d, e2) a right-handed frame.
void normal_frame(const Vec &p_d, Vec &r_e0, Vec &r_e2) {
    const Vec ref = std::fabs(p_d.x) < 0.9 ? Vec{ 1.0, 0.0, 0.0 } : Vec{ 0.0, 0.0, 1.0 };
    r_e0 = unit(cross(p_d, ref));
    r_e2 = cross(r_e0, p_d);
}

// The images of x and z under the smallest rotation taking +y to the unit p_up (Rodrigues): the identity for +y. Below
// the horizon it goes through half a turn about x first (taking +y to -y), so 1 + cos never comes near 0.
void up_frame(const Vec &p_up, Vec &r_x, Vec &r_z) {
    const bool below = p_up.y < 0.0;
    const Vec from{ 0.0, below ? -1.0 : 1.0, 0.0 };
    const Vec axis = cross(from, p_up);
    const double c = below ? -p_up.y : p_up.y;
    auto turned = [&](const Vec &p_a) {
        const Vec a = below ? Vec{ p_a.x, -p_a.y, -p_a.z } : p_a;
        return a + cross(axis, a) + cross(axis, cross(axis, a)) * (1.0 / (1.0 + c));
    };
    r_x = turned(Vec{ 1.0, 0.0, 0.0 });
    r_z = turned(Vec{ 0.0, 0.0, 1.0 });
}

// p_axis turned by p_angle towards the normal direction at azimuth p_azimuth.
Vec turn(const Vec &p_axis, double p_angle, double p_azimuth) {
    Vec e0, e2;
    normal_frame(p_axis, e0, e2);
    const Vec n = e0 * std::cos(p_azimuth) + e2 * std::sin(p_azimuth);
    return unit(p_axis * std::cos(p_angle) + n * std::sin(p_angle));
}

using ExactNoise::hash3;
using ExactNoise::unit_of;

constexpr double TAU = 6.283185307179586;
constexpr double GOLDEN_ANGLE = 2.399963229728653;
// Below half a voxel of radius a branch never meshes, even at LOD 0.
constexpr double MIN_RADIUS = 0.5;
// A stem of order n is split at most MAX_REFINE - n levels deep: the trunk in up to 8 segments, limbs 4, twigs 2.
constexpr int MAX_REFINE = 3;
constexpr int MAX_ORDER = 2;
// Rotation components smaller than this are 0 (keeps every product of the program in the exact domain).
constexpr double ROTATION_FLUSH = 0x1p-24;

// A record's rotation from its local axes in absolute coordinates (the rows: absolute to local).
void rotation_rows(float r_rows[9], const Vec &p_x, const Vec &p_y, const Vec &p_z) {
    const Vec rows[3] = { p_x, p_y, p_z };
    for (int i = 0; i < 3; i++) {
        const double c[3] = { rows[i].x, rows[i].y, rows[i].z };
        for (int j = 0; j < 3; j++) {
            r_rows[3 * i + j] = std::fabs(c[j]) < ROTATION_FLUSH ? 0.0f : float(c[j]);
        }
    }
}

// A tree's records before their anchor and rotation: the grower's and the species' bounds (tree_bounds) alike.
TableRecord branch_record(const TreeTypes &p_types, double p_h, double p_ra, double p_slope, double p_k) {
    TableRecord rec;
    rec.def = uint8_t(p_types.branch);
    rec.op = Combine::SMOOTH_UNION;
    rec.set_k(float(p_k));
    rec.params[0] = float(p_h);
    rec.params[1] = float(p_ra);
    rec.params[2] = float(p_slope);
    return rec;
}

TableRecord crown_record(const TreeTypes &p_types, double p_r, double p_k, uint32_t p_salt) {
    TableRecord rec;
    rec.def = uint8_t(p_types.crown);
    rec.op = Combine::SMOOTH_UNION;
    rec.set_k(float(p_k));
    rec.params[0] = float(p_r);
    rec.params[1] = p_types.crown_amp;
    rec.salt = p_salt;
    return rec;
}

TableRecord cone_record(const TreeTypes &p_types, double p_r, double p_height, double p_k, uint32_t p_salt) {
    const double slant = std::sqrt(p_r * p_r + p_height * p_height);
    TableRecord rec;
    rec.def = uint8_t(p_types.cone);
    rec.op = Combine::SMOOTH_UNION;
    rec.set_k(float(p_k));
    rec.params[0] = float(p_r);
    rec.params[1] = float(p_height);
    rec.params[2] = float(p_height / slant); // cos of the half angle
    rec.params[3] = float(p_r / slant); // sin
    rec.params[4] = p_types.crown_amp;
    rec.salt = p_salt;
    return rec;
}

// The scale a tree's hashed factor stays under (Grower::grow: 0.8 + 0.4 * unit_of).
constexpr double SCALE_MAX = 1.2;

struct Stem {
    Vec p0; // voxels from the tree's base, whole voxels
    Vec dir; // unit
    double length = 0.0;
    double r0 = 0.0;
    int order = 0;
    uint32_t path = 0; // a hash of the path from the root
    uint32_t seq = 0; // the order it was found in (ties)
};

struct Thicker {
    bool operator()(const Stem &p_a, const Stem &p_b) const { return p_a.r0 < p_b.r0 || (p_a.r0 == p_b.r0 && p_a.seq > p_b.seq); }
};

struct Joint {
    Vec p;
    double t = 0.0; // along the stem's axis, 0 at its base, 1 at its tip
};

struct Sphere {
    Vec c;
    double r = 0.0;
};

class Grower {
public:
    Grower(const TreeSpecies &p_species, const VoxelPos &p_base, uint32_t p_seed, double p_voxel, const TreeTypes &p_types,
            std::vector<TableRecord> &r_out, std::vector<TreeStem> *r_stems, const Vec &p_up, bool p_hull_only) :
            s(p_species), base(p_base), seed(p_seed), voxel(p_voxel), types(p_types), out(r_out), stems(r_stems), up(p_up), hull_only(p_hull_only) {}

    void grow() {
        const ExactNoise::UVec3 h = hash3(seed, 0x7EEu, 0u);
        scale = 0.8 + 0.4 * unit_of(h.x);
        Stem trunk;
        trunk.r0 = v(s.trunk_radius);
        trunk.p0 = snap(up * -trunk.r0); // sunk by its radius: the flare blends it into the ground
        const double lean = s.lean * unit_of(h.y);
        trunk.dir = turn(up, lean, TAU * unit_of(h.z));
        trunk.length = v(s.height) + trunk.r0;
        trunk.path = hash3(seed, 0x7EEu, 1u).x;
        left = s.budget - (s.cone_tiers > 0 ? s.cone_tiers : s.hull_spheres);
        heap.push(trunk);
        while (!heap.empty() && left > 0) {
            const Stem stem = heap.top();
            heap.pop();
            grow_stem(stem);
        }
        for (size_t i = 0; i < hull.size(); i++) {
            add_crown(hull[i], hash3(seed, 0x4011u, uint32_t(i)).x);
        }
        if (s.cone_tiers > 0) {
            add_tiers(trunk_joints);
        }
    }

private:
    const TreeSpecies &s;
    const VoxelPos base;
    const uint32_t seed;
    const double voxel;
    const TreeTypes &types;
    std::vector<TableRecord> &out;
    std::vector<TreeStem> *stems;
    const Vec up; // unit: the trunk's axis before its lean, the cone tiers' axis
    const bool hull_only; // the trunk's polyline for the hull, nothing else (grow_tree)
    double scale = 1.0;
    int left = 0;
    uint32_t seq = 0;
    std::vector<Sphere> hull;
    std::vector<Joint> trunk_joints; // a conifer's tiers close the tree, like the hull
    std::priority_queue<Stem, std::vector<Stem>, Thicker> heap;

    // Metres to voxels, at this tree's scale.
    double v(double p_metres) const { return p_metres * scale / voxel; }

    // A child stem before it is grown: where it forks off its parent (axis parameter), its radius, direction
    // and length.
    struct Fork {
        double t = 0.0;
        double r = 0.0;
        double angle = 0.0;
        double azimuth = 0.0;
        double length = 0.0;
        uint32_t path = 0;
    };

    // The children of p_stem and its tip radius, by the pipe model: the children's r^2 plus the tip's is at most
    // the base's (the children shrink together if needed); none thinner than MIN_RADIUS.
    std::vector<Fork> forks_of(const Stem &p_stem, double &r_end) const {
        std::vector<Fork> forks;
        const int n = p_stem.order == 0 ? s.limbs : (p_stem.order == 1 ? s.twigs : 0);
        const double t_lo = p_stem.order == 0 ? s.crown_base : 0.3;
        const double t_hi = p_stem.order == 0 ? 0.95 : 0.9;
        const double phase = TAU * unit_of(hash3(p_stem.path, 0xF0u, 0u).x);
        for (int i = 0; i < n; i++) {
            const ExactNoise::UVec3 h = hash3(p_stem.path, uint32_t(i + 1), 0xC41Du);
            Fork f;
            f.t = t_lo + (t_hi - t_lo) * (double(i) + 0.2 + 0.6 * unit_of(h.x)) / double(n);
            f.r = s.child_ratio * p_stem.r0 * (1.0 - 0.5 * f.t);
            f.angle = s.limb_angle * (0.8 + 0.4 * unit_of(h.y));
            f.azimuth = phase + GOLDEN_ANGLE * double(i);
            const double jitter = 0.85 + 0.3 * unit_of(h.z);
            f.length = p_stem.order == 0 ? v(s.crown_radius) * (1.1 - 0.5 * (f.t - s.crown_base) / (1.0 - s.crown_base)) * jitter
                                         : s.twig_length * p_stem.length * (1.0 - 0.4 * f.t) * jitter;
            f.path = hash3(p_stem.path, uint32_t(i + 1), 0x9A7Eu).x;
            forks.push_back(f);
        }
        auto keep_thick = [&]() {
            std::vector<Fork> kept;
            double sum = 0.0;
            for (const Fork &f : forks) {
                if (f.r >= MIN_RADIUS) {
                    kept.push_back(f);
                    sum += f.r * f.r;
                }
            }
            forks.swap(kept);
            return sum;
        };
        double sum = keep_thick();
        const double cap = (1.0 - s.tip_ratio * s.tip_ratio) * p_stem.r0 * p_stem.r0;
        if (sum > cap) {
            for (Fork &f : forks) {
                f.r *= std::sqrt(cap / sum);
            }
            sum = keep_thick();
        }
        r_end = MAX(MIN(std::sqrt(MAX(p_stem.r0 * p_stem.r0 - sum, 0.0)), s.end_ratio * p_stem.r0), s.tip_ratio * p_stem.r0);
        return forks;
    }

    void grow_stem(const Stem &p_stem) {
        TreeStem info;
        info.order = p_stem.order;
        info.r0 = p_stem.r0;
        double r_end = 0.0;
        const std::vector<Fork> forks = forks_of(p_stem, r_end);
        info.r_end = r_end;
        auto radius_at = [&](double p_t) { return p_stem.r0 + (r_end - p_stem.r0) * p_t; };

        // The polyline: the axis split at moved midpoints (endpoints never move).
        const Joint root{ p_stem.p0, 0.0 };
        const Joint tip{ snap(p_stem.p0 + p_stem.dir * p_stem.length), 1.0 };
        std::vector<Joint> joints{ root };
        refine(p_stem, radius_at, root, tip, 1u, MAX(MAX_REFINE - p_stem.order, 0), joints);

        if (p_stem.order == 0 && s.cone_tiers > 0) {
            trunk_joints = joints;
        } else if (p_stem.order == 0) {
            // The hull: spheres on the trunk's polyline, above the crown base, the last one on the trunk's tip (it
            // would poke out above the crown).
            for (int i = 0; i < s.hull_spheres; i++) {
                const double f = s.hull_spheres > 1 ? double(i) / double(s.hull_spheres - 1) : 0.0;
                const double t = s.crown_base + (1.0 - s.crown_base) * double(i + 1) / double(s.hull_spheres);
                hull.push_back(Sphere{ point_at(joints, t), v(s.crown_radius * s.hull_ratio) * (1.0 + (s.hull_taper - 1.0) * f) });
            }
        }
        if (hull_only) {
            // The hull (and the tiers) are emitted after the loop from what is set now, in both modes; the budget reserved
            // them, so nothing grown here could have cut them.
            if (stems) {
                stems->push_back(info);
            }
            return;
        }
        for (size_t k = 0; k + 1 < joints.size() && left > 0; k++) {
            const double ra = radius_at(joints[k].t);
            const double rb = radius_at(joints[k + 1].t);
            if (inside_hull(joints[k].p, ra, joints[k + 1].p, rb)) {
                continue; // never shows: the hull covers it
            }
            const double k_smooth = k > 0 ? 0.0 : p_stem.r0 * (p_stem.order == 0 ? s.flare : s.fork_smooth);
            add_branch(joints[k].p, joints[k + 1].p, ra, rb, k_smooth);
        }
        for (const Fork &f : forks) {
            Stem child;
            child.order = p_stem.order + 1;
            child.p0 = snap(point_at(joints, f.t));
            child.dir = turn(direction_at(joints, f.t), f.angle, f.azimuth);
            child.length = f.length;
            child.r0 = MIN(f.r, radius_at(f.t)); // never thicker than the parent where it forks
            child.path = f.path;
            child.seq = ++seq;
            if (child.r0 >= MIN_RADIUS) {
                info.children.push_back(child.r0);
                heap.push(child);
            }
        }
        if (info.children.empty() && p_stem.order > 0 && left > 0) {
            // A leaf stem: a clump at its tip.
            const double r = v(s.crown_radius * s.clump_ratio) * (0.8 + 0.4 * unit_of(hash3(p_stem.path, 0xC1u, 0u).x));
            const double reach = r + double(types.crown_dent); // its own bumps reach out by up to crown_dent
            if (!inside_hull(tip.p, reach, tip.p, reach)) {
                add_crown(Sphere{ tip.p, r }, hash3(p_stem.path, 0xC1u, 1u).x);
            }
        }
        if (stems) {
            stems->push_back(info);
        }
    }

    template <typename R>
    void refine(const Stem &p_stem, const R &p_radius_at, Joint p_a, Joint p_b, uint32_t p_node, int p_levels, std::vector<Joint> &r_joints) const {
        const Vec seg = p_b.p - p_a.p;
        const double l = length(seg);
        if (p_levels == 0 || l <= s.segment_ratio * MAX(p_radius_at(p_a.t), p_radius_at(p_b.t))) {
            r_joints.push_back(p_b);
            return;
        }
        const ExactNoise::UVec3 h = hash3(p_stem.path, p_node, 0xBE4Du);
        const Vec d = seg * (1.0 / l);
        Vec e0, e2;
        normal_frame(d, e0, e2);
        const double azimuth = TAU * unit_of(h.x);
        const double offset = (p_stem.order == 0 ? s.trunk_tortuosity : s.tortuosity) * l * (2.0 * unit_of(h.y) - 1.0);
        const Joint m{ snap((p_a.p + p_b.p) * 0.5 + (e0 * std::cos(azimuth) + e2 * std::sin(azimuth)) * offset), 0.5 * (p_a.t + p_b.t) };
        refine(p_stem, p_radius_at, p_a, m, 2u * p_node, p_levels - 1, r_joints);
        refine(p_stem, p_radius_at, m, p_b, 2u * p_node + 1u, p_levels - 1, r_joints);
    }

    // The polyline's point and direction at axis parameter p_t.
    static size_t span_at(const std::vector<Joint> &p_joints, double p_t) {
        size_t k = 0;
        while (k + 2 < p_joints.size() && p_joints[k + 1].t < p_t) {
            k++;
        }
        return k;
    }
    static Vec point_at(const std::vector<Joint> &p_joints, double p_t) {
        const size_t k = span_at(p_joints, p_t);
        const Joint &a = p_joints[k];
        const Joint &b = p_joints[k + 1];
        const double w = (p_t - a.t) / (b.t - a.t);
        return a.p + (b.p - a.p) * w;
    }
    static Vec direction_at(const std::vector<Joint> &p_joints, double p_t) {
        const size_t k = span_at(p_joints, p_t);
        return unit(p_joints[k + 1].p - p_joints[k].p);
    }

    // Whether two balls lie inside one hull sphere, even where its bumps dent it most: then so does everything
    // between them (a tapered segment is inside the convex hull of its end balls).
    bool inside_hull(const Vec &p_a, double p_ra, const Vec &p_b, double p_rb) const {
        for (const Sphere &h : hull) {
            const double inner = h.r - double(types.crown_dent);
            if (length(p_a - h.c) + p_ra <= inner && length(p_b - h.c) + p_rb <= inner) {
                return true;
            }
        }
        return false;
    }

    VoxelPos anchor_of(const Vec &p_p) const { return base + VoxelPos(std::llround(p_p.x), std::llround(p_p.y), std::llround(p_p.z)); }

    void add_branch(const Vec &p_a, const Vec &p_b, double p_ra, double p_rb, double p_k) {
        const Vec seg = p_b - p_a;
        const double h = length(seg);
        if (h < 1.0) {
            return; // both ends snapped to one voxel
        }
        TableRecord rec = branch_record(types, h, p_ra, (p_rb - p_ra) / h, p_k);
        rec.anchor = anchor_of(p_a);
        const Vec d = seg * (1.0 / h);
        Vec e0, e2;
        normal_frame(d, e0, e2);
        rotation_rows(rec.rotation, e0, d, e2);
        out.push_back(rec);
        left--;
    }

    // A conifer's crown: cone_tiers vertical cones on the trunk, each twice the spacing high (they overlap by half),
    // the top one's apex at the trunk's tip, radii from crown_radius down to hull_taper times it.
    void add_tiers(const std::vector<Joint> &p_joints) {
        const int n = s.cone_tiers;
        Vec tier_x, tier_z;
        up_frame(up, tier_x, tier_z); // the tiers stand along up
        const double length = v(s.height) * (1.0 - s.crown_base);
        const double spacing = length / double(n + 1);
        for (int i = 0; i < n; i++) {
            const double f = n > 1 ? double(i) / double(n - 1) : 0.0;
            const double t = s.crown_base + (1.0 - s.crown_base) * double(i) / double(n + 1);
            const ExactNoise::UVec3 h = hash3(seed, 0x7135u, uint32_t(i));
            const double r = v(s.crown_radius) * (1.0 + (s.hull_taper - 1.0) * f) * (0.9 + 0.2 * unit_of(h.x));
            TableRecord rec = cone_record(types, r, 2.0 * spacing, s.crown_smooth * r, h.y);
            rec.anchor = anchor_of(snap(point_at(p_joints, t)));
            rotation_rows(rec.rotation, tier_x, up, tier_z);
            out.push_back(rec);
            left--;
        }
    }

    // p_salt: the noise salt, a hash of the part's path (not of how many parts came before it).
    void add_crown(const Sphere &p_sphere, uint32_t p_salt) {
        TableRecord rec = crown_record(types, p_sphere.r, s.crown_smooth * p_sphere.r, p_salt);
        rec.anchor = anchor_of(snap(p_sphere.c));
        out.push_back(rec);
        left--;
    }
};

} // namespace

const std::vector<TreeSpecies> &tree_species() {
    static const std::vector<TreeSpecies> species = []() {
        std::vector<TreeSpecies> list;
        list.push_back(TreeSpecies()); // a broadleaf: a round crown on a short trunk
        TreeSpecies slender; // taller, a narrow crown from low down
        slender.height = 18.0;
        slender.trunk_radius = 0.25;
        slender.crown_base = 0.3;
        slender.crown_radius = 2.4;
        slender.hull_ratio = 0.6;
        slender.hull_spheres = 3;
        slender.limbs = 8;
        slender.twigs = 1;
        slender.limb_angle = 1.15;
        list.push_back(slender);
        TreeSpecies conifer; // stacked cones from low down, the trunk bare below them
        conifer.height = 20.0;
        conifer.trunk_radius = 0.28;
        conifer.lean = 0.04;
        conifer.crown_base = 0.2;
        conifer.crown_radius = 3.0;
        conifer.cone_tiers = 4;
        conifer.hull_taper = 0.45;
        conifer.limbs = 0;
        conifer.crown_smooth = 0.15;
        list.push_back(conifer);
        return list;
    }();
    return species;
}

int crown_k(double p_voxel) { return ExactNoise::lattice_k(CROWN_WAVELENGTH_M, p_voxel); }

TreeTypes crown_types(int p_branch, int p_crown, const TableDef &p_crown_def, int p_cone) {
    // The rock's bounds at radius 1 and amp 1: half size 1 + reach, L = 1 + slope (per unit of amp).
    const float unit[8] = { 1.0f, 1.0f, 0, 0, 0, 0, 0, 0 };
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_crown_def.bounds(unit, lo, hi, lipschitz);
    TreeTypes t;
    t.branch = p_branch;
    t.crown = p_crown;
    t.cone = p_cone;
    const double slope = double(lipschitz) - 1.0;
    t.crown_amp = slope > 0.0 ? float(CROWN_BUMP / slope) : 0.0f;
    t.crown_dent = float(double(t.crown_amp) * (double(hi.x) - 1.0));
    return t;
}

TreeTypes add_tree_types(DefLibrary &r_library, double p_voxel) {
    const int branch = r_library.add(make_branch_type(), Material::BARK);
    const int crown = r_library.add(make_rock_type(crown_k(p_voxel), CROWN_SEED), Material::LEAVES);
    const int cone = r_library.add(make_cone_type(crown_k(p_voxel), CROWN_SEED), Material::LEAVES);
    return crown_types(branch, crown, r_library[crown], cone);
}

void up_rotation(const UpAxis &p_up, float r_rows[9]) {
    const Vec up{ p_up[0], p_up[1], p_up[2] };
    Vec x, z;
    up_frame(up, x, z);
    rotation_rows(r_rows, x, up, z);
}

void grow_tree(const TreeSpecies &p_species, const VoxelPos &p_base, const UpAxis &p_up, uint32_t p_seed, double p_voxel, const TreeTypes &p_types,
        std::vector<TableRecord> &r_out, std::vector<TreeStem> *r_stems, bool p_hull_only) {
    Grower(p_species, p_base, p_seed, p_voxel, p_types, r_out, r_stems, Vec{ p_up[0], p_up[1], p_up[2] }, p_hull_only).grow();
}

TreeBounds tree_bounds(const TreeSpecies &p_species, double p_voxel, const TreeTypes &p_types) {
    const TreeSpecies &s = p_species;
    TreeBounds b;
    const double v = SCALE_MAX / p_voxel; // metres to voxels at the largest scale
    // The trunk's base segment: radius v(trunk_radius) with the flare's k; its height at most the trunk's length (v(height)
    // + r0, the sinking) plus a voxel at each snapped end. Every other segment is thinner (the taper, the pipe model) with
    // a smaller k (the fork smooth of a thinner child, or none).
    const double r0 = v * s.trunk_radius * TREE_BOUND_PAD;
    b.trunk = branch_record(p_types, v * s.height + r0 + 2.0, r0, 0.0, s.flare * r0);
    b.has_clump = s.limbs > 0;
    if (b.has_clump) {
        // A leaf stem's clump: v(crown_radius * clump_ratio) times its hashed 0.8-1.2 (grow_stem).
        const double r = v * s.crown_radius * s.clump_ratio * 1.2 * TREE_BOUND_PAD;
        b.clump = crown_record(p_types, r, s.crown_smooth * r, 0u);
    }
    const double taper = MAX(1.0, s.hull_taper);
    if (s.cone_tiers > 0) {
        // A tier: v(crown_radius) times the taper and its hashed 0.9-1.1, twice the tiers' spacing high (add_tiers).
        const double r = v * s.crown_radius * taper * 1.1 * TREE_BOUND_PAD;
        b.hull = cone_record(p_types, r, 2.0 * v * s.height * (1.0 - s.crown_base) / double(s.cone_tiers + 1), s.crown_smooth * r, 0u);
    } else {
        const double r = v * s.crown_radius * s.hull_ratio * taper * TREE_BOUND_PAD;
        b.hull = crown_record(p_types, r, s.crown_smooth * r, 0u);
    }
    return b;
}

int tree_hull_only_lod(const TreeSpecies &p_species, double p_voxel, const DefLibrary &p_library, const RecordFinish &p_finish, const TreeTypes &p_types) {
    TreeBounds b = tree_bounds(p_species, p_voxel, p_types);
    finish_table_record(p_library, p_finish, b.trunk);
    int lod = int(b.trunk.max_lod);
    if (b.has_clump) {
        finish_table_record(p_library, p_finish, b.clump);
        lod = MAX(lod, int(b.clump.max_lod));
    }
    return lod; // the hull is what stays
}

int tree_species_of(uint32_t p_variant) {
    return int(p_variant % uint32_t(tree_species().size()));
}

void set_forest_scatter(CellScatter &r_scatter, double p_voxel, const DefLibrary &p_library, const RecordFinish &p_finish, const TreeTypes &p_types,
        int64_t p_max_extent) {
    std::vector<int> hull_lod;
    for (const TreeSpecies &species : tree_species()) {
        const TreeBounds b = tree_bounds(species, p_voxel, p_types);
        bool fits = true;
        for (const TableRecord *rec : { &b.trunk, &b.clump, &b.hull }) {
            const bool counts = rec != &b.clump || b.has_clump;
            const String error = counts ? record_fit_error(p_library[rec->def], *rec, p_max_extent) : String();
            if (!error.is_empty()) {
                ERR_PRINT("set_forest_scatter: a tree of a species may not fit the table, it never grows hull-only: " + error);
                fits = false;
            }
        }
        hull_lod.push_back(fits ? tree_hull_only_lod(species, p_voxel, p_library, p_finish, p_types) : int(TableRecord::NEVER_DROPPED));
    }
    r_scatter.salt = TREE_SITE_SALT;
    r_scatter.generate = [p_types, p_voxel, hull_lod](const Site &p_site, const UpAxis &p_up, int p_lod, std::vector<TableRecord> &r_out) {
        const int species = tree_species_of(p_site.variant);
        grow_tree(tree_species()[species], p_site.base, p_up, p_site.seed, p_voxel, p_types, r_out, nullptr, p_lod >= hull_lod[species]);
    };
    r_scatter.classes = uint32_t(tree_species().size()); // class_index = tree_species_of
    r_scatter.profile_key = "trees:" + key_bits(p_voxel) + key_bits(tree_species().size()) + key_bits(p_types.branch) + key_bits(p_types.crown) + key_bits(p_types.cone);
}

} // namespace Archetype
