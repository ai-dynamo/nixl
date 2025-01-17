#ifndef __TRANSFER_REQUEST_H_
#define __TRANSFER_REQUEST_H_

// Contains pointers to corresponding backend engine and its handler, and populated
// and verified DescLists, and other state and metadata needed for a NIXL transfer
class nixlXferReqH {
    private:
        nixlBackendEngine* engine;
        nixlBackendReqH*   backendHandle;

        nixl_meta_dlist_t* initiatorDescs;
        nixl_meta_dlist_t* targetDescs;

        std::string        remoteAgent;
        std::string        notifMsg;

        nixl_xfer_op_t     backendOp;
        nixl_xfer_state_t  state;

    public:
        inline nixlXferReqH() {
            initiatorDescs = nullptr;
            targetDescs    = nullptr;
            engine         = nullptr;
            backendHandle  = nullptr;
        }

        inline ~nixlXferReqH() {
            // delete checks for nullptr itself
            delete initiatorDescs;
            delete targetDescs;
            if (backendHandle != nullptr)
                engine->releaseReqH(backendHandle);
        }

    friend class nixlAgent;
};

class nixlXferSideH {
    private:
        nixl_meta_dlist_t* descs;

        nixlBackendEngine* engine;
        std::string        remoteAgent;
        bool               isLocal;

    public:
        inline nixlXferSideH() {
            engine = nullptr;
        }

        inline ~nixlXferSideH() {
            delete descs;
        }

    friend class nixlAgent;
};

#endif
