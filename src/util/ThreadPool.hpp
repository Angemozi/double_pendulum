// =============================================================================
// ThreadPool.hpp
// -----------------------------------------------------------------------------
// A minimal persistent thread pool used to parallelize the PPO gradient update
// across CPU cores.
//
// WHY PERSISTENT (not std::async per task):
//   A PPO update issues hundreds of small parallel regions (one per minibatch x
//   epoch). Spawning threads each time would cost more than the work itself.
//   This pool spins up N worker threads ONCE and re-dispatches a tiny task into
//   them via a generation counter + condition variable. Workers stay warm.
//
// API:
//   pool.run(fn)  -> invokes fn(workerId) exactly once on each of the N workers
//                    and blocks until all have finished. The caller (main) thread
//                    participates only as the coordinator, so N == core budget.
//
// DETERMINISM:
//   The pool guarantees every worker id 0..N-1 runs exactly once per run(). The
//   PPO code assigns a FIXED, contiguous data shard to each worker id and reduces
//   partial results in worker-id order, so results are bit-reproducible for a
//   fixed worker count (see PPOAgent::update).
// =============================================================================
#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <cstddef>

namespace dp::util {

class ThreadPool {
public:
    explicit ThreadPool(int numWorkers) : numWorkers_(numWorkers) {
        workers_.reserve(static_cast<std::size_t>(numWorkers_));
        for (int id = 0; id < numWorkers_; ++id)
            workers_.emplace_back([this, id] { workerLoop(id); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        startCv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const noexcept { return numWorkers_; }

    // Run fn(workerId) on every worker once; block until all complete.
    void run(const std::function<void(int)>& fn) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = &fn;
            remaining_ = numWorkers_;
            ++generation_;
        }
        startCv_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        doneCv_.wait(lock, [this] { return remaining_ == 0; });
        task_ = nullptr;
    }

private:
    void workerLoop(int id) {
        std::uint64_t lastGen = 0;
        while (true) {
            const std::function<void(int)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                startCv_.wait(lock, [this, &lastGen] { return generation_ != lastGen; });
                lastGen = generation_;
                if (stop_) return;
                job = task_;
            }
            if (job) (*job)(id);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_ == 0) doneCv_.notify_one();
            }
        }
    }

    int numWorkers_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable startCv_;
    std::condition_variable doneCv_;
    const std::function<void(int)>* task_ = nullptr;
    int           remaining_  = 0;
    std::uint64_t generation_ = 0;
    bool          stop_       = false;
};

} // namespace dp::util
