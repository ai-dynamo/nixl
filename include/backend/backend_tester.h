#ifndef __BACKEND_TESTER_H_
#define __BACKEND_TESTER_H_

#include <mutex>
#include <string>
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include "utils/sys/nixl_time.h"

#include "backend_engine.h"

//Class for unit tests to be able to access backend engine functions directly
class nixlBackendTester {

    private:
        nixlBackendEngine *engine;

    public:
        nixlBackendTester(nixlBackendEngine* &new_engine) {
            engine = new_engine;
        }
        ~nixlBackendTester() {
        }

        nixl_backend_t getType () { 
            return engine->getType(); 
        }
        bool getInitErr () { 
            return engine->getInitErr();
        }
        
        bool supportsLocal () { 
            return engine->supportsLocal();
        }
        bool supportsRemote () { 
            return engine->supportsRemote();
        }
        bool supportsNotif () { 
            return engine->supportsNotif();
        }
        bool supportsProgTh () { 
            return engine->supportsProgTh();
        }
        nixl_status_t registerMem (const nixlStringDesc &mem,
                                   const nixl_mem_t &nixl_mem,
                                   nixlBackendMD* &out) { 
            return engine->registerMem(mem, nixl_mem, out);
        }
        void deregisterMem (nixlBackendMD* meta) {
            engine->deregisterMem(meta);
        }
        std::string getPublicData (const nixlBackendMD* meta) {
            return engine->getPublicData(meta);
        }
        std::string getConnInfo() {
            return engine->getConnInfo();
        }
        nixl_status_t loadRemoteConnInfo (const std::string &remote_agent,
                                          const std::string &remote_conn_info){
            return engine->loadRemoteConnInfo(remote_agent, remote_conn_info);
        }
        nixl_status_t connect(const std::string &remote_agent){
            return engine->connect(remote_agent);
        }
        nixl_status_t disconnect(const std::string &remote_agent){
            return engine->disconnect(remote_agent);
        }
        nixl_status_t unloadMD (nixlBackendMD* input){
            return engine->unloadMD(input);
        }
        nixl_status_t loadRemoteMD (const nixlStringDesc &input,
                                    const nixl_mem_t &nixl_mem,
                                    const std::string &remote_agent,
                                    nixlBackendMD* &output){
            return engine->loadRemoteMD(input, nixl_mem, remote_agent, output);
        }
        nixl_status_t loadLocalMD (nixlBackendMD* input,
                                   nixlBackendMD* &output) {
            return engine->loadLocalMD(input, output);
        }
        nixl_xfer_state_t postXfer (const nixlDescList<nixlMetaDesc> &local,
                                    const nixlDescList<nixlMetaDesc> &remote,
                                    const nixl_xfer_op_t &operation,
                                    const std::string &remote_agent,
                                    const std::string &notif_msg,
                                    nixlBackendReqH* &handle){
            return engine->postXfer(local, remote, operation, remote_agent, notif_msg, handle);
        }
        nixl_xfer_state_t checkXfer(nixlBackendReqH* handle){
            return engine->checkXfer(handle);
        }
        void releaseReqH(nixlBackendReqH* handle){
            engine->releaseReqH(handle);
        }
        int getNotifs(notif_list_t &notif_list) { 
            return engine->getNotifs(notif_list);
        }
        nixl_status_t genNotif(const std::string &remote_agent, const std::string &msg) {
            return engine->genNotif(remote_agent, msg);
        }
        int progress() { 
            return engine->progress(); 
        }
};
#endif
