#pragma once

#include "tree_archetype.hpp"

#include <vector>

// The three tree species the engine had built in before the species file (worlds/species.json holds them to the bit,
// test_biome_pick.cpp): a broadleaf, a slender tree and a conifer. Tests grow them without a world.
inline const std::vector<Archetype::TreeSpecies> &tree_species() {
    static const std::vector<Archetype::TreeSpecies> species = []() {
        std::vector<Archetype::TreeSpecies> list;
        list.push_back(Archetype::TreeSpecies()); // a broadleaf: a round crown on a short trunk
        Archetype::TreeSpecies slender; // taller, a narrow crown from low down
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
        Archetype::TreeSpecies conifer; // stacked cones from low down, the trunk bare below them
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
