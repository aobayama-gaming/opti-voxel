// The streamed sources' walk batches (instance_stream.hpp, Settings::walk_batch): walk keys computed together.
#include "instance_stream.hpp"

#include "debug_log.hpp"

namespace Archetype {

void InstanceStream::take_walks(std::vector<std::unique_ptr<Job>> &r_batch) {
    // The queue's next walk jobs of the same source, in rank order, looking a few batches deep (the others stay for the
    // other tasks). A walk key has no input: its first job computes it.
    const ScatterSource *source = r_batch[0]->source;
    size_t looked = 0;
    const size_t most = walk_batch.load();
    for (auto it = jobs.begin(); it != jobs.end() && r_batch.size() < most && looked < WALK_LOOK * most; looked++) {
        if (it->second->key.walk() && it->second->source == source) {
            if (!executor->more()) {
                break; // the executor's budget: one job per step
            }
            r_batch.push_back(std::move(it->second));
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
}

void InstanceStream::run_walks(std::vector<std::unique_ptr<Job>> &r_batch) {
    const uint64_t t0 = now_usec();
    std::vector<SourceKey> keys;
    std::vector<SourceContent *> outs;
    for (std::unique_ptr<Job> &job : r_batch) {
        job->out = std::make_shared<SourceContent>();
        keys.push_back(job->key);
        outs.push_back(job->out.get());
    }
    r_batch[0]->source->walk_tiles(keys.data(), keys.size(), outs.data());
    const uint64_t each = (now_usec() - t0) / r_batch.size();
    for (std::unique_ptr<Job> &job : r_batch) {
        job->usec = each; // their compute time shared
    }
}

} // namespace Archetype
