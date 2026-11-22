#pragma once

#include "test_common.hpp"

#include "sdf_cascade.hpp"
#include "species.hpp"
#include "world_file.hpp"

#include <vector>

// A test's world when it names none: the template ground (worlds/ground.biomes.json) and today's species
// (worlds/species.json), the files the bare worlds name. A world names its ground: an SDFCascade without biomes is
// refused (SDFCascade::attach_climate).

// The template ground's climate table (WorldFile::ground_table).
inline ov::Value ground_table() {
    ov::Value table;
    CHECK(WorldFile::ground_table(table) == ov::str());
    return table;
}

// Today's species: three trees (broadleaf, slender, conifer: tree_species()), a boulder and a pillar.
inline std::vector<Archetype::Species> default_species() {
    std::vector<Archetype::Species> species;
    CHECK(WorldFile::load_species(WorldFile::SPECIES, species) == ov::str());
    return species;
}

// An SDFCascade on the template ground with today's species, until a test gives it another climate.
struct GroundSDF : SDFCascade {
    GroundSDF() {
        climate_data = ground_table();
        set_species(default_species());
    }
};
