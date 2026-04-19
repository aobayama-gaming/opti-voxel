#include "mesher_pool.hpp"


void MesherPool::start(
        int num_workers,
        int lod_numbers,
        RID p_scenario,
        SDFBase* p_sdf,
        ChunkParam p_chunk_param
    ){

    scenario = p_scenario;
    sdf=p_sdf;
    chunk_param = p_chunk_param;

    workers.clear();
    workers.resize(num_workers, nullptr);

    for(int i=0;i<num_workers;i++){

        workers[i] = memnew(CPUMesher);

        workers[i]->start(sdf,chunk_param,scenario);

    }

    chunk_queue.resize(lod_numbers, nullptr);

    for (int i = 0; i < lod_numbers; ++i) {
        chunk_queue[i] = memnew(LODHashSet);
    }

}

void MesherPool::send_chunk(LODHashSet p_chunks){

    for (const Vector3i &chunk_pos : p_chunks) {

    const int lod = ChunkMath::get_lod(chunk_pos);
    if (lod < 0 || lod >= static_cast<int>(chunk_queue.size())) {
        continue;
    }

    LODHashSet *q = chunk_queue[lod];
    if (q) {
        q->insert(chunk_pos);
    }
}

}

bool MesherPool::_pop_queue(Vector3i &new_chunk) {
    for (LODHashSet *queue : chunk_queue) {
        if (queue == nullptr || queue->is_empty()) {
            continue;
        }

        LODHashSet::Iterator it = queue->begin();
        if (!it) {
            continue;
        }

        Vector3i chunk_pos = *it;
        queue->remove(it);
        new_chunk = chunk_pos;
        return true;
    }

    return false;
}

void MesherPool::process_queue(){

    for(CPUMesher* worker : workers){


        if(worker->is_available()){

            Vector3i new_chunk;

            if(_pop_queue(new_chunk)){

                UtilityFunctions::print("launch worker");

                worker->launch_pipeline(new_chunk);
            }

        }

    }

}

bool MesherPool::get_results(ChunkHashMap &output) {

    bool has_any_result = false;

    for (CPUMesher *worker : workers) {
        if (worker == nullptr) {
            continue;
        }

        ChunkOutput result;
        Vector3i chunk_id;

        if (worker->get_result(chunk_id,result)) {
            output.insert(chunk_id,result);
            has_any_result = true;
        }
    }

    return has_any_result;
}