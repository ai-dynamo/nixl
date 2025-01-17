#ifndef _NIXL_PARAMS_H
#define _NIXL_PARAMS_H

#include <string>
#include <cstdint>
#include "backend/backend_engine.h"
#include "nixl_types.h"

// Per Agent configuration information, such as if progress thread should be used.
// Other configs such as assigned IP/port or device access can be added.
class nixlAgentConfig {
    private:

        // Determines if progres thread is used or not
        bool     useProgThread;

    public:

        /*
         * Progress thread frequency knob (in us)
         * The progress thread is calling sched_yield to avoid blocking a core
         * If pthrDelay time is less than sched_yield time - option has no effect
         * Otherwise pthread will be calling sched_yield untill the specified
         * amount of time has past.
         */
        uint64_t pthrDelay;

        // Important configs such as useProgThread must be given and can't be changed.
        nixlAgentConfig(const bool use_prog_thread, const uint64_t pthr_delay_us=0) {
            this->useProgThread = use_prog_thread;
            this->pthrDelay     = pthr_delay_us;
        }
        nixlAgentConfig(const nixlAgentConfig &cfg) = default;
        ~nixlAgentConfig() = default;

    friend class nixlAgent;
};

// Example backend initialization data for UCX.
// For now UCX autodetects devices, later might add hints.
class nixlUcxInitParams : public nixlBackendInitParams
{
    public:

        /*
         * Restrict the list of used UCX devices
         * empty vector instructs UCX to use all
         * supported devices found on the node
         */
        std::vector<std::string> devices;

        inline nixl_backend_t getType() const { return UCX; }

};

class nixlGdsInitParams : public nixlBackendInitParams
{
    public:
        std::vector<std::string> mount_targets;
        inline nixl_backend_t getType() const { return GPUDIRECTIO;}
};

#endif
