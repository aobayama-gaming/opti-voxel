#include "test_common.hpp"

#include "forest_generator.hpp"

#include <algorithm>
#include <cmath>
#include <string>

using namespace Archetype;

namespace {

// The types trees use with their materials, registered as SDFTable::build_table does.
struct ForestLibrary {
    DefLibrary lib;
    TreeTypes types;
    explicit ForestLibrary(double p_voxel) {
        const int branch = lib.add(make_branch_type(), Material::BARK);
        const int crown = lib.add(make_rock_type(crown_k(p_voxel), CROWN_SEED), Material::LEAVES);
        const int cone = lib.add(make_cone_type(crown_k(p_voxel), CROWN_SEED), Material::LEAVES);
        types = crown_types(branch, crown, lib[crown], cone);
    }
};

double quantile(std::vector<double> p_v, double p_q) {
    if (p_v.empty()) {
        return 0.0;
    }
    std::sort(p_v.begin(), p_v.end());
    return p_v[size_t(p_q * double(p_v.size() - 1))];
}

double distance(const double *p_a, const double *p_b) {
    return std::sqrt((p_a[0] - p_b[0]) * (p_a[0] - p_b[0]) + (p_a[1] - p_b[1]) * (p_a[1] - p_b[1]) + (p_a[2] - p_b[2]) * (p_a[2] - p_b[2]));
}

} // namespace

// What the far field gives up by reading a tree's species class instead of its own parts: per tree and material, the
// relative error of its volume, of its centroid (against its spread) and of its ellipsoid's largest semi-axis; and,
// what a far blob draws, the sum over many trees.
TEST_CASE("Forest: a tree's class moments stand for its own, the sum over many trees closely") {
    const double voxel = 0.05;
    const ForestLibrary fl(voxel);
    const TreeClasses classes(fl.lib, fl.types, voxel);
    for (int sp = 0; sp < int(tree_species().size()); sp++) {
        std::vector<double> mass_err, centre_err, axis_err;
        double true_sum[2] = {}, class_sum[2] = {};
        const uint32_t materials[2] = { Material::BARK, Material::LEAVES };
        for (uint32_t i = 0; i < 400; i++) {
            TreeSite site;
            site.seed = ExactNoise::hash3(i, uint32_t(sp), 0x7E57u).x;
            site.species = sp;
            std::vector<TableRecord> parts;
            grow_tree(tree_species()[sp], VoxelPos(), UP_Y, site.seed, voxel, fl.types, parts);
            const MaterialMoments truth = span_moments(fl.lib, parts.data(), uint32_t(parts.size()), VoxelPos());
            const MaterialMoments guess = classes.tree(site, UP_Y);
            for (int m = 0; m < 2; m++) {
                const Moments *t = truth.find(materials[m]);
                const Moments *g = guess.find(materials[m]);
                CHECK((t == nullptr) == (g == nullptr));
                if (!t || !g) {
                    continue;
                }
                true_sum[m] += t->m0;
                class_sum[m] += g->m0;
                mass_err.push_back(std::fabs(g->m0 / t->m0 - 1.0));
                const double spread = std::sqrt((t->s[0] + t->s[1] + t->s[2]) / t->m0);
                centre_err.push_back(distance(g->c, t->c) / spread);
                double ta[3], ga[3];
                t->semi_axes(ta);
                g->semi_axes(ga);
                axis_err.push_back(std::fabs(ga[2] / ta[2] - 1.0));
            }
        }
        MESSAGE("species ", sp, ": volume error median ", quantile(mass_err, 0.5), " p95 ", quantile(mass_err, 0.95), "; centroid / spread median ",
                quantile(centre_err, 0.5), " p95 ", quantile(centre_err, 0.95), "; largest semi-axis median ", quantile(axis_err, 0.5), " p95 ",
                quantile(axis_err, 0.95), "; summed volume bark ", class_sum[0] / true_sum[0], " leaves ", class_sum[1] / true_sum[1]);
        // Measured (feature/forest-generator, 0.05 m): volume median 0.4-1.1%, p95 8-12%; centroid median 0.08-0.13 of the
        // spread; largest semi-axis median 0.2-0.9%; sums within 0.5%.
        CHECK(quantile(mass_err, 0.5) < 0.03);
        CHECK(quantile(mass_err, 0.95) < 0.2);
        CHECK(quantile(centre_err, 0.5) < 0.2);
        CHECK(quantile(axis_err, 0.5) < 0.02);
        for (int m = 0; m < 2; m++) {
            if (true_sum[m] > 0.0) {
                CHECK(std::fabs(class_sum[m] / true_sum[m] - 1.0) < 0.02); // what a blob over many trees weighs
            }
        }
    }
}

TEST_CASE("Forest: a tree's class moments turn with its up and grow with its scale") {
    const double voxel = 0.05;
    const ForestLibrary fl(voxel);
    const TreeClasses classes(fl.lib, fl.types, voxel);
    TreeSite site;
    site.seed = 0x1234u;
    const MaterialMoments upright = classes.tree(site, UP_Y);
    const MaterialMoments sideways = classes.tree(site, UpAxis{ 1.0, 0.0, 0.0 });
    const Moments *a = upright.find(Material::LEAVES);
    const Moments *b = sideways.find(Material::LEAVES);
    CHECK(a != nullptr);
    CHECK(b != nullptr);
    if (a && b) {
        CHECK(a->c[1] > 10.0 * std::fabs(a->c[0])); // the crown above the base
        CHECK(b->c[0] == doctest::Approx(a->c[1]).epsilon(1e-6)); // turned onto +x
        CHECK(b->m0 == doctest::Approx(a->m0).epsilon(1e-9));
        const double s = tree_scale(site.seed);
        CHECK(a->m0 == doctest::Approx(classes.mean(0, TreeClasses::size_bucket(s)).find(Material::LEAVES)->m0 * s * s * s).epsilon(1e-9));
    }
}

// The forest is a pure function of its cells: two halves of a box give the trees of the whole, keyed by distinct cells.
TEST_CASE("Forest generator: the trees of two halves are the trees of the whole") {
    const VoxelPos origin(int64_t(1) << 36, 2500, -77777);
    const double tau = 6.283185307179586;
    const FieldFn hills = [origin, tau](const VoxelPos &p_v) {
        const double amp = 80.0, wave = 1800.0;
        const double x = double(p_v.x - origin.x), z = double(p_v.z - origin.z);
        const double w = double(p_v.y - origin.y) - amp * std::sin(tau * x / wave) * std::cos(tau * z / wave);
        return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave) * std::cos(tau * z / wave)), 1.0f,
            float(amp * tau / wave * std::sin(tau * x / wave) * std::sin(tau * z / wave)), float(w) };
    };
    ForestGenerator forest;
    forest.side = 90;
    forest.zone = VoxelBox{ origin - VoxelPos(1500, 600, 1500), origin + VoxelPos(1500, 600, 1500), true };
    std::vector<CellSite> whole, halves;
    forest.trees(VoxelBox{ origin - VoxelPos(5000, 5000, 5000), origin + VoxelPos(5000, 5000, 5000), true }, hills, whole);
    forest.trees(VoxelBox{ origin - VoxelPos(5000, 5000, 5000), origin + VoxelPos(-1, 5000, 5000), true }, hills, halves);
    forest.trees(VoxelBox{ origin - VoxelPos(0, 5000, 5000), origin + VoxelPos(5000, 5000, 5000), true }, hills, halves);
    auto order = [](const CellSite &p_a, const CellSite &p_b) {
        const VoxelPos &a = p_a.cell, &b = p_b.cell;
        return a.z != b.z ? a.z < b.z : (a.y != b.y ? a.y < b.y : a.x < b.x);
    };
    std::sort(halves.begin(), halves.end(), order);
    MESSAGE("forest: ", whole.size(), " trees whole, ", halves.size(), " from two halves");
    CHECK(whole.size() > 500);
    CHECK(whole.size() == halves.size());
    if (whole.size() == halves.size()) {
        for (size_t i = 0; i < whole.size(); i++) {
            CHECK(whole[i].cell == halves[i].cell);
            CHECK(whole[i].site.base == halves[i].site.base);
            CHECK(whole[i].site.seed == halves[i].site.seed);
            CHECK(forest.zone.contains(whole[i].site.base));
            if (i > 0) {
                CHECK(order(whole[i - 1], whole[i])); // distinct keys, in cell order
            }
        }
    }
}
