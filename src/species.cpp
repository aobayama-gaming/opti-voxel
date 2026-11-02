#include "species.hpp"

#include "ov/format.hpp"

#include <cmath>
#include <initializer_list>
#include <utility>

namespace Archetype {

namespace {

// TreeSpecies' keys: its doubles and its ints.
const std::pair<const char *, double TreeSpecies::*> TREE_DOUBLES[] = {
    { "height", &TreeSpecies::height },
    { "trunk_radius", &TreeSpecies::trunk_radius },
    { "lean", &TreeSpecies::lean },
    { "crown_base", &TreeSpecies::crown_base },
    { "crown_radius", &TreeSpecies::crown_radius },
    { "hull_ratio", &TreeSpecies::hull_ratio },
    { "hull_taper", &TreeSpecies::hull_taper },
    { "limb_angle", &TreeSpecies::limb_angle },
    { "child_ratio", &TreeSpecies::child_ratio },
    { "end_ratio", &TreeSpecies::end_ratio },
    { "tip_ratio", &TreeSpecies::tip_ratio },
    { "twig_length", &TreeSpecies::twig_length },
    { "tortuosity", &TreeSpecies::tortuosity },
    { "trunk_tortuosity", &TreeSpecies::trunk_tortuosity },
    { "segment_ratio", &TreeSpecies::segment_ratio },
    { "clump_ratio", &TreeSpecies::clump_ratio },
    { "flare", &TreeSpecies::flare },
    { "fork_smooth", &TreeSpecies::fork_smooth },
    { "crown_smooth", &TreeSpecies::crown_smooth },
};
const std::pair<const char *, int TreeSpecies::*> TREE_INTS[] = {
    { "hull_spheres", &TreeSpecies::hull_spheres },
    { "cone_tiers", &TreeSpecies::cone_tiers },
    { "limbs", &TreeSpecies::limbs },
    { "twigs", &TreeSpecies::twigs },
    { "budget", &TreeSpecies::budget },
};

// One entry of the list (p_where names it in the error).
ov::str parse_one(const ov::Value &p_d, const ov::str &p_where, Species &r_s) {
    if (!p_d.is_object() || !p_d["name"].is_string() || !p_d["kind"].is_string()) {
        return p_where + ": a species is an object with a \"name\" and a \"kind\" (tree, rock, pillar).";
    }
    r_s.name = p_d["name"].as_str();
    const ov::str kind = p_d["kind"].as_str();
    const ov::str where = p_where + " '" + r_s.name + "'";
    if (kind == "tree") {
        r_s.kind = Species::TREE;
        for (const std::pair<ov::str, ov::Value> &e : p_d.entries()) {
            if (e.first == "name" || e.first == "kind" || e.first == "comment") {
                continue;
            }
            bool known = false;
            for (const auto &[key, member] : TREE_DOUBLES) {
                if (e.first == key) {
                    known = true;
                    if (!e.second.is_number() || !(e.second.as_double() >= 0.0) || !std::isfinite(e.second.as_double())) {
                        return where + ": '" + e.first + "' is a number >= 0.";
                    }
                    r_s.tree.*member = e.second.as_double();
                }
            }
            for (const auto &[key, member] : TREE_INTS) {
                if (e.first == key) {
                    known = true;
                    const double v = e.second.is_number() ? e.second.as_double() : -1.0;
                    if (!(v >= 0.0 && v <= 4096.0 && v == std::floor(v))) {
                        return where + ": '" + e.first + "' is a whole number in [0, 4096].";
                    }
                    r_s.tree.*member = int(v);
                }
            }
            if (!known) {
                return where + ": unknown key '" + e.first + "'.";
            }
        }
        return r_s.tree.height > 0.0 && r_s.tree.trunk_radius > 0.0 && r_s.tree.crown_radius > 0.0 ? ov::str() : where + ": a tree's height, trunk and crown are > 0.";
    }
    if (kind != "rock" && kind != "pillar") {
        return where + ": the kind is tree, rock or pillar.";
    }
    r_s.kind = Species::ROCK;
    r_s.rock.pillar = kind == "pillar";
    for (const std::pair<ov::str, ov::Value> &e : p_d.entries()) {
        if (e.first != "name" && e.first != "kind" && e.first != "comment" && e.first != "radius_m") {
            return where + ": unknown key '" + e.first + "'.";
        }
    }
    if (p_d.has("radius_m")) {
        const ov::Value &r = p_d["radius_m"];
        if (!(r.is_array() && r.size() == 2 && r[size_t(0)].is_number() && r[size_t(1)].is_number())) {
            return where + ": 'radius_m' is [lo, hi] in metres.";
        }
        r_s.rock.radius_lo_m = r[size_t(0)].as_double();
        r_s.rock.radius_hi_m = r[size_t(1)].as_double();
    }
    return r_s.rock.radius_lo_m > 0.0 && r_s.rock.radius_hi_m >= r_s.rock.radius_lo_m && std::isfinite(r_s.rock.radius_hi_m) ? ov::str()
                                                                                                                             : where + ": its radius is 0 < lo <= hi.";
}

} // namespace

std::vector<Species> default_species() {
    std::vector<Species> list;
    const char *const TREES[] = { "broadleaf", "slender", "conifer" };
    for (size_t i = 0; i < tree_species().size(); i++) {
        Species s;
        s.name = i < std::size(TREES) ? TREES[i] : ov::format("tree%d", int64_t(i));
        s.tree = tree_species()[i];
        list.push_back(s);
    }
    for (const bool pillar : { false, true }) {
        Species s;
        s.name = pillar ? "pillar" : "boulder";
        s.kind = Species::ROCK;
        s.rock.pillar = pillar;
        list.push_back(s);
    }
    return list;
}

ov::str parse_species(const ov::Value &p_list, std::vector<Species> &r_out) {
    if (!p_list.is_array()) {
        return "species: a list of species.";
    }
    std::vector<Species> out;
    for (size_t i = 0; i < p_list.size(); i++) {
        Species s;
        const ov::str err = parse_one(p_list[i], ov::format("species %d", int64_t(i)), s);
        if (!err.empty()) {
            return err;
        }
        for (const Species &o : out) {
            if (o.name == s.name) {
                return "species: '" + s.name + "' is named twice.";
            }
        }
        out.push_back(s);
    }
    r_out = out;
    return ov::str();
}

std::vector<const Species *> species_of(const std::vector<Species> &p_all, Species::Kind p_kind) {
    std::vector<const Species *> out;
    for (const Species &s : p_all) {
        if (s.kind == p_kind) {
            out.push_back(&s);
        }
    }
    return out;
}

std::string species_key(const std::vector<const Species *> &p_species) {
    // Each species' own kind's parameters, as bits (key_bits: two that differ in the last bit don't share a profile).
    std::string key;
    for (const Species *s : p_species) {
        key += std::string(s->name.c_str()) + ":";
        if (s->kind == Species::TREE) {
            for (const auto &[name, member] : TREE_DOUBLES) {
                key += key_bits(s->tree.*member);
            }
            for (const auto &[name, member] : TREE_INTS) {
                key += key_bits(s->tree.*member);
            }
        } else {
            key += key_bits(s->rock.pillar) + key_bits(s->rock.radius_lo_m) + key_bits(s->rock.radius_hi_m);
        }
        key += ";";
    }
    return key;
}

} // namespace Archetype
