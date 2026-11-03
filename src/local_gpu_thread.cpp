#include "local_gpu_thread.hpp"

#include "trace.hpp"

std::unique_ptr<LocalGpuThread> LocalGpuThread::create() {
    std::unique_ptr<LocalGpuThread> thread(new LocalGpuThread());
    thread->worker = ov::executor().start_worker("local_gpu", *thread);
    return thread;
}

LocalGpuThread::~LocalGpuThread() {
    if (worker) {
        worker->stop();
    }
}

void LocalGpuThread::run(const std::function<void(ov::gpu::Device *)> &p_job) {
    if (!ov::executor().threaded()) { // the inline backend: enter ran on this thread, the device is this thread's
        p_job(device.get());
        return;
    }
    Job job;
    job.fn = &p_job;
    std::unique_lock<std::mutex> lock(mutex);
    jobs.push_back(&job);
    lock.unlock();
    worker->wake();
    lock.lock();
    job_done.wait(lock, [&job]() { return job.done; });
}

void LocalGpuThread::enter() {
    OV_TRACE("local_gpu", "create_device");
    device = ov::gpu::create_local_device();
}

bool LocalGpuThread::step() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!jobs.empty()) {
        Job *job = jobs.front();
        jobs.pop_front();
        lock.unlock();
        (*job->fn)(device.get());
        lock.lock();
        job->done = true;
        job_done.notify_all();
    }
    return false;
}

void LocalGpuThread::exit() {
    device.reset(); // its parked objects with it (Device::drop_parked)
}
