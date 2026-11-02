#pragma once

#include "tree_archetype.hpp"

#include "ov/str.hpp"
#include "ov/value.hpp"

#include <vector>

// The world's species (a world file's "species"; default_species without one): what its biomes grow (their
// "instances", Climate::InstanceShare), each named. A tree is a TreeSpecies (its keys, metres and ratios, each optional
// at TreeSpecies' default); a rock is a boulder or a pillar of a hashed radius in [radius_m[0], radius_m[1]).
//   { "name": "conifer", "kind": "tree", "height": 20, "crown_radius": 3, "cone_tiers": 4, ... }
//   { "name": "boulder", "kind": "rock", "radius_m": [1, 4] }, { "name": "pillar", "kind": "pillar", "radius_m": [1, 4] }
// The trees are the forest source's classes, the rocks the scatter's, in list order.
namespace Archetype {

struct RockSpecies {
    bool pillar = false; // a capsule standing on the ground, else a rock sunk 0.4 of its radius
    double radius_lo_m = 1.0;
    double radius_hi_m = 4.0;
};

struct Species {
    enum Kind {
        TREE,
        ROCK,
    };
    ov::str name;
    Kind kind = TREE;
    TreeSpecies tree;
    RockSpecies rock;
};

// Today's: the three trees of tree_species() (broadleaf, slender, conifer), a boulder and a pillar.
std::vector<Species> default_species();
// p_list (a world file's "species") in r_out: empty, or the first error (r_out untouched).
ov::str parse_species(const ov::Value &p_list, std::vector<Species> &r_out);
// The species of kind p_kind, in list order (a source's classes).
std::vector<const Species *> species_of(const std::vector<Species> &p_all, Species::Kind p_kind);
// A key of p_species' parameters (SourceProfile's registry: sources with the same key share one profile).
ov::str species_key(const std::vector<const Species *> &p_species);

} // namespace Archetype
