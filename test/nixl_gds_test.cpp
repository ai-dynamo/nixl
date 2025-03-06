#include <iostream>
#include <string>
#include <algorithm>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>


#define NUM_TRANSFERS 1
#define SIZE 1024


int main(int argc, char *argv[])
{
	nixl_status_t		ret = NIXL_SUCCESS;
	void			*addr[NUM_TRANSFERS];
	std::string		role;
	int			status = 0;
	int			i;
	int			fd[NUM_TRANSFERS];

	nixlAgentConfig		cfg(true);
	nixlGdsInitParams	params;
	nixlStringDesc		buf[NUM_TRANSFERS];
	nixlStringDesc		ftrans[NUM_TRANSFERS];
	nixlBackendEngine	*gds;

	nixl_reg_dlist_t	vram_for_gds(VRAM_SEG);
	nixl_reg_dlist_t	file_for_gds(FILE_SEG);
	nixlXferReqH		*treq;
	std::string		names[NUM_TRANSFERS];

	std::cout << "Starting Agent for " << "GDS Test Agent" << "\n";
	nixlAgent	agent("GDSTester", cfg);
	gds	      = agent.createBackend(&params);

	for (i = 0; i < NUM_TRANSFERS; i++) {
		cudaMalloc((void **)&addr[i], SIZE);
		cudaMemset(addr[i], 'A', SIZE);
		std::cout << "Allocating for src buffer : "
			  << addr[i] << ","
			  << "Setting to 0xbb "
			  << std::endl;
		names[i] = std::string("/mnt/gds/test_file") + "_" +
			std::to_string(i);
		std::cout << "Opened File " << names[i] << std::endl;
		fd[i] = open(names[i].c_str(), O_RDWR|O_CREAT);
		if (fd[i] < 0) {
		   std::cerr<<"Open call failed to open file\n";
		   return 1;
		}
		buf[i].addr   = (uintptr_t)(addr[i]);
		buf[i].len    = SIZE;
		buf[i].devId  = 0;
		vram_for_gds.addDesc(buf[i]);

		ftrans[i].addr  = i * SIZE; // this is offset
		ftrans[i].len   = SIZE;
		ftrans[i].devId = fd[i];
		file_for_gds.addDesc(ftrans[i]);
	}
	agent.registerMem(file_for_gds, gds);
	agent.registerMem(vram_for_gds, gds);

	nixl_xfer_dlist_t vram_for_gds_list = vram_for_gds.trim();
	nixl_xfer_dlist_t file_for_gds_list = file_for_gds.trim();

	ret = agent.createXferReq(vram_for_gds_list, file_for_gds_list,
				  "GDSTester", "", NIXL_WRITE, treq);
	if (ret != NIXL_SUCCESS) {
		std::cerr << "Error creating transfer request\n";
		exit(-1);
	}

	std::cout << " Post the request with GDS backend\n ";
	status = agent.postXferReq(treq);
	std::cout << " GDS File IO has been posted\n";
	std::cout << " Waiting for completion\n";

	while (status != NIXL_XFER_DONE) {
	    status = agent.getXferStatus(treq);
	    assert(status != NIXL_XFER_ERR);
	}
	std::cout <<" Completed writing data using GDS backend\n";
	agent.invalidateXferReq(treq);

	std::cout <<"Cleanup.. \n";
	agent.deregisterMem(vram_for_gds, gds);
	agent.deregisterMem(file_for_gds, gds);



	return 0;
}

