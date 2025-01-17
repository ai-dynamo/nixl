/** NVIDIA Inference Xfer Library */
#ifndef _NIXL_H
#define _NIXL_H

#include "nixl_types.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

// Main transfer object
class nixlAgent {
    private:
        nixlAgentData* data;

    public:

        /*** Initialization and Registering Methods ***/

        // Populates agent name and device metadata
        nixlAgent (const std::string &name, const nixlAgentConfig &cfg);
        ~nixlAgent ();

        // Instantiate BackendEngine objects, based on corresponding params
        nixlBackendEngine* createBackend (nixlBackendInitParams* params);
        // Register with the backend and populate memory_section
        nixl_status_t registerMem (const nixl_reg_dlist_t &descs,
                                   nixlBackendEngine* backend);
        // Deregister and remove from memory section
        nixl_status_t deregisterMem (const nixl_reg_dlist_t &descs,
                                     nixlBackendEngine* backend);

        // Make connection proactively, instead of at transfer time
        nixl_status_t makeConnection (const std::string &remote_agent);


        /*** Transfer Request Handling ***/

        // Creates a transfer request, with automatic backend selection.
        nixl_status_t createXferReq (const nixl_xfer_dlist_t &local_descs,
                                     const nixl_xfer_dlist_t &remote_descs,
                                     const std::string &remote_agent,
                                     const std::string &notif_msg,
                                     const nixl_xfer_op_t &operation,
                                     nixlXferReqH* &req_handle);

        // Submit a transfer request, which populates the req async handler.
        nixl_xfer_state_t postXferReq (nixlXferReqH* req);

        // Check the status of transfer requests
        nixl_xfer_state_t getXferStatus (nixlXferReqH* req);

        // Invalidate transfer request if we no longer need it.
        // Will also abort a running transfer.
        void invalidateXferReq (nixlXferReqH* req);


        /*** Alternative method to create transfer handle manually ***/

        // User can ask for backend chosen for a XferReq to use it for prepXferSide.
        nixlBackendEngine* getXferBackend(nixlXferReqH* req_handle);

        // Prepares descriptors for one side of a transfer with given backend.
        // Empty string for remote_agent means it's local side.
        nixl_status_t prepXferSide (const nixl_xfer_dlist_t &descs,
                                    const std::string &remote_agent,
                                    const nixlBackendEngine* backend,
                                    nixlXferSideH* &side_handle);

        // Makes a transfer request from already prepared side transfer handles.
        nixl_status_t makeXferReq (const nixlXferSideH* local_side,
                                   const std::vector<int> &local_indices,
                                   const nixlXferSideH* remote_side,
                                   const std::vector<int> &remote_indices,
                                   const std::string &notif_msg,
                                   const nixl_xfer_op_t &operation,
                                   nixlXferReqH* &req_handle);

        void invalidateXferSide (nixlXferSideH* side_handle);

        /*** Notification Handling ***/

        // Add entries to the passed received notifications list (can be
        // non-empty), and return number of added entries, or -1 if there was
        // an error. Elements are released within the Agent after this call.
        int getNotifs (nixl_notifs_t &notif_map);

        // Generate a notification, not bound to a transfer, e.g., for control.
        // Can be used after the remote metadata is exchanged. Will be received
        // in notif list. Nixl will choose a backend if null is passed.
        nixl_status_t genNotif (const std::string &remote_agent,
                                const std::string &msg,
                                nixlBackendEngine* backend = nullptr);


        /*** Metadata handling through side channel ***/

        // Get nixl_metadata for this agent. Empty string means error.
        // The std::string used for serialized MD can have \0 values.
        std::string getLocalMD () const;

        // Load other agent's metadata and unpack it internally.
        // Returns the found agent name in metadata, or "" in case of error.
        std::string loadRemoteMD (const std::string &remote_metadata);

        // Invalidate the remote section information cached locally
        nixl_status_t invalidateRemoteMD (const std::string &remote_agent);
};

#endif
