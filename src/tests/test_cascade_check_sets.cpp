#include "test_common.hpp"

#include "cascade_check_sets.hpp"

using namespace CascadeCheckSets;

TEST_CASE("CascadeCheckSets::parse_sets: labels with spaces and empty items, the generic bit") {
    std::vector<SpikeSet> sets;
    CHECK(parse_sets("4,4c", 0, sets) == ov::str());
    CHECK(sets == std::vector<SpikeSet>{ SpikeSet::L4, SpikeSet::L4C });
    CHECK(parse_sets(" 4 ,\t4c,\n", 0, sets) == ov::str()); // whitespace round a label, a trailing comma
    CHECK(sets == std::vector<SpikeSet>{ SpikeSet::L4, SpikeSet::L4C });
    CHECK(parse_sets("4s", GENERIC, sets) == ov::str());
    CHECK(sets == std::vector<SpikeSet>{ SpikeSet(int(SpikeSet::L4S) | GENERIC) });
    CHECK(set_label(SpikeSet(int(SpikeSet::L4S) | GENERIC)) == "4s generic");
}

TEST_CASE("CascadeCheckSets::parse_sets: an unknown label or no set at all is refused, never a check of nothing") {
    std::vector<SpikeSet> sets;
    CHECK(ov::contains(parse_sets("4,4x", 0, sets), "unknown set '4x'"));
    CHECK(ov::contains(parse_sets("4;4c", 0, sets), "unknown set '4;4c'"));
    CHECK(ov::contains(parse_sets("", 0, sets), "no set"));
    CHECK(ov::contains(parse_sets(" , ", 0, sets), "no set"));
    CHECK(sets.empty());
}

TEST_CASE("CascadeCheckSets::configure_variant: a world file that doesn't load fails the set, never the default world") {
    SDFCascade sdf;
    ChunkParam param;
    ov::str error;
    CHECK(configure_variant(sdf, SpikeSet::L4, "res://worlds/does_not_exist.json", param, &error) == 0);
    CHECK(ov::contains(error, "does_not_exist"));
    CHECK(configure_variant(sdf, SpikeSet::L4, "res://worlds/lods25.world.json", param, &error) == 25);
    CHECK(error == ov::str());
}
