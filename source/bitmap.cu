/*
 Copyright 2016 Kashtanova Anna Viktorovna
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "bitmap.cuh"
#include "helper.cuh"
#include <cassert>
int chooseDevice()
{
	int count = 0;
	uint8_t max_major = 0;
	uint8_t max_minor = 0;
	int device = -1;
	checkError( cudaGetDeviceCount(&count) );
	cudaDeviceProp properties;

	for ( int i = 0; i < count; ++i )
	{
		if( properties.major > max_major )
		{
			max_major = properties.major;
			device = i;
		}else if( properties.major == max_major && properties.major > max_minor) {
			max_minor = properties.minor;
			device = i;
		}
	}
	if( device >= 0 ){
		checkError( cudaGetDeviceProperties(&properties, device) );
		std::cout<<"Device is integrated: ";
		( properties.integrated ) ? std::cout<<"yes.\n" : std::cout<<"no.\n";
		std::cout<<"Device name: "<<properties.name<<std::endl;
		std::cout<<"Compute device capabilities are : "<<properties.major
				<<"."<<properties.minor<<std::endl;
		std::cout<<"Grid dimention: ("
				<<properties.maxGridSize[0]<<", "
				<<properties.maxGridSize[1]<<", "
				<<properties.maxGridSize[2]<<")"<<std::endl;
		std::cout<<"Max. number of threads per block: "
				<<properties.maxThreadsPerBlock<<std::endl;
		std::cout<<"Block dimention: ("
				<<properties.maxThreadsDim[0]<<", "
				<<properties.maxThreadsDim[1]<<", "
				<<properties.maxThreadsDim[2]<<")"<<std::endl;
		std::cout<<"Max size of shared memory per block: "
				<<properties.sharedMemPerBlock<<" bytes."<<std::endl;
		std::cout<<"-------------------------------------"<<std::endl;

		return device;
	}else{
		return 0;
	}
}
