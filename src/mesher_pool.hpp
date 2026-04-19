#pragma once

#include <godot_cpp/classes/node.hpp>
#include "CPU_mesher.hpp"
#include <vector>
#include "lod_structures.hpp"
#include "chunk_memory.hpp"

using namespace godot;

class MesherPool : public Object {

private:

    std::vector<CPUMesher *> workers;

    std::vector<LODHashSet *> chunk_queue;

    LODHashSet working_generation;


    RID scenario;
    SDFBase* sdf;
    ChunkParam chunk_param;

    bool _pop_queue(Vector3i &new_chunk);

public:

    void start(
        int num_workers,
        int lod_numbers,
        RID p_scenario,
        SDFBase* p_sdf,
        ChunkParam p_chunk_param
    );

    void send_chunk(LODHashSet p_chunks);

    void process_queue();

    bool get_results(VoxelState &output);


};