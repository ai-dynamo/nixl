/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GPUDIRECT_TCPXO_LOCKFREE_QUEUE_H_
#define GPUDIRECT_TCPXO_LOCKFREE_QUEUE_H_

#include <deque>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace tcpxo {

/*
 * A multi-producer, multi-consumer queue
 *
 * A lockfree queue implementation is a problem with many opinions. For a lock-full implementation,
 * we should probably use a mutex and an stl datastructure. If the lock-based implementation is too
 * slow and a lockfree implementation is preferred, consider using:
 *   1. Boost's lockfree queue:
 *   https://www.boost.org/doc/libs/latest/doc/html/lockfree/examples.html
 *   2. Moodycamel's lockfree queue: https://github.com/cameron314/concurrentqueue
 *   3. Implementing the Michael-Scott lockfree queue:
 *   https://www.cs.rochester.edu/u/scott/papers/1996_PODC_queues.pdf
 */
template<typename T> class MPMCQueue {
public:
    void
    Enqueue(T &&item) {
        absl::MutexLock lock(&queue_mutex_);
        queue_.push_back(std::move(item));
    }

    absl::StatusOr<T>
    TryDequeue() {
        absl::MutexLock lock(&queue_mutex_);
        if (queue_.empty()) {
            return absl::OutOfRangeError("queue empty");
        }
        T item = std::move(queue_.front());
        queue_.pop_front();
        return item;
    }

    bool
    IsEmpty() {
        absl::MutexLock lock(&queue_mutex_);
        return queue_.empty();
    }

    size_t
    size() {
        absl::MutexLock lock(&queue_mutex_);
        return queue_.size();
    }

private:
    absl::Mutex queue_mutex_;
    std::deque<T> queue_ ABSL_GUARDED_BY(queue_mutex_);
};

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_LOCKFREE_QUEUE_H_
