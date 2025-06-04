#DOCA plugin unit test

Unit test application for the DOCA backend plugin. It enables a GPU to GPU communications between initiator and target using the GPUDirect Async Kernel-Initiated (GDAKI) technology through the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) library.

Initiator and target communicate over the network using the DOCA plugin, so both processes should run on a system with a GPU. Initiator and target exchange a sequence of metadata and notifications for the setup then initiator posts twice the same transfer request with 32 buffers, with a value for the first transfer and updated value for the second transfer.

It's possible to execute in two modes:
- Stream attached: when creating a transfer request, the application can specify a CUDA stream where the transfer request (executed by a CUDA kernel implemented within the plugin) must be posted. Useful in situations where data is prepared in a GPU CUDA kernel and right after, asynchronously, it can be sent by the GPU without requiring any CPU synchronization.
- Stream pool: at transfer request creation time, no CUDA stream is provided by the app. The DOCA plugin can still be used as it uses an internal pool of CUDA streams used in a round-robin fashion.

Please note in this unit test, data used for transfers is allocated in GPU memory. In real-world applications this is not mandatory: the DOCA plugin can send/receive data from the CUDA kernel if it resides in CPU memory.

## System requirements

Please look at the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) programming guide to properly configure youy system for the correct execution of this application.

## Execute

The following instructions assume DOCA libraries and NIXL libraries are set in LD_LIBRARY_PATH environment variable.

Target should start first. A command line example:

```
./nixl_doca_stream_test target <initiator IP address> <connection port> <stream mode ('attached' or 'pool')>
```

To start the initiator:

```
./nixl_doca_stream_test initiator <target IP address> <connection port> <stream mode>
```

## Output

To ensure everything works correctly as expected, tagert should print out 32 lines like the following ones for the first transfer:

```
>>>>>>> CUDA target kernel, buffer 0, all bytes received! val=187
>>>>>>> CUDA target kernel, buffer 1, all bytes received! val=187
...
>>>>>>> CUDA target kernel, buffer 30, all bytes received! val=187
>>>>>>> CUDA target kernel, buffer 31, all bytes received! val=187
```

and another set of 32 lines for the second tranfer:

```
>>>>>>> CUDA target kernel, buffer 0, all bytes received! val=188
>>>>>>> CUDA target kernel, buffer 1, all bytes received! val=188
...
>>>>>>> CUDA target kernel, buffer 30, all bytes received! val=188
>>>>>>> CUDA target kernel, buffer 31, all bytes received! val=188
```
