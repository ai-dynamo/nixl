#ifndef __AGENT_DATA_H_
#define __AGENT_DATA_H_

#include "mem_section.h"

class nixlAgentData {
    private:
        std::string                                  name;
        nixlAgentConfig                              config;

        std::map<nixl_backend_t, nixlBackendEngine*> nixlBackendEngines;
        std::map<nixl_backend_t, std::string>        connMD; // Local info

        nixlLocalSection                             memorySection;

        std::map<std::string, nixlRemoteSection*>    remoteSections;
        std::map<std::string, backend_set_t>         remoteBackends;

        nixlAgentData(const std::string &name, const nixlAgentConfig &cfg);
        ~nixlAgentData();

    friend class nixlAgent;
};

#endif
