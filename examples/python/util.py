def wait_for_transfer_completion(init_agent, target_agent, xfer_handle, uuid):
    """Wait for both initiator and target to complete transfer."""
    target_done = False
    init_done = False

    while (not init_done) or (not target_done):
        if not init_done:
            state = init_agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                raise RuntimeError("Transfer got to Error state.")
            if state == "DONE":
                init_done = True
                print("Initiator done")

        if (not target_done) and target_agent.check_remote_xfer_done("initiator", uuid):
            target_done = True
            print("Target done")
