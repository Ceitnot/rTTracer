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
#include "acceleration.cuh"
#include "algorithms.cuh"
#include "helper.cuh"
__host__ __device__
Boundaries::Boundaries(){
    for (uint8_t i = 0; i < nNormals; ++i)
        distances[i][0] = kInfinity, distances[i][1] = -kInfinity;
}
__host__ __device__
bool Boundaries::hit(Ray &ray, float (&precomPuted)[2][7], uint8_t *planeIndex) const 
{
	//here ray.t is closest plane hitPoint and ray.tNearest is the other side planes hit point
	for (uint8_t i = 0; i < nNormals; ++i) {

        float tn = (distances[i][0] - precomPuted[0][i]) / precomPuted[1][i];
        float tf = (distances[i][1] - precomPuted[0][i]) / precomPuted[1][i];
        if (precomPuted[1][i] < 0) algorithms::swap(tn, tf);

        if (tn > ray.t) ray.t = tn, *planeIndex = i;
        if (tf < ray.tNearest) ray.tNearest = tf;
        if (ray.t > ray.tNearest) return false;
    }
	return true;
}
__host__ __device__
const float& Boundaries::at(size_t planePairNumber, size_t farOrClose) const
{// must throw an exception
	return distances[planePairNumber][farOrClose];
}
#ifdef __ACCELERATION2
template <class T>
__global__ void compute_min_max(T * elements, compare comp, size_t size, T* result)
{
	__shared__ T tmp [ threadsInblock ];
	int tid  = threadIdx.x + blockDim.x*blockIdx.x;
	tmp[threadIdx.x] = kInfinity;
	if(tid < size){
		tmp[ threadIdx.x ] = elements[ tid ];
	__syncthreads();
	for( int i = 2; i <= threadsInblock; i*=2 )
	{
		if( !(threadIdx.x % i) ){
			if( comp(tmp[threadIdx.x + i/2], tmp[threadIdx.x] )
					&& (threadIdx.x + i) <= threadsInblock  )
			{
				swap(tmp[threadIdx.x], tmp[threadIdx.x + i/2]);
			}
		}
	}
	__syncthreads();
	if( threadIdx.x == 0 )
		result[ blockIdx.x ] = tmp[0];
	}
}

float acceleratedMinMax(float *gpu_elements, size_t size, compare& dev_compare, compare cpu_compare)
{

		float *result;

		int sresult = (size + threadsInblock - 1)/threadsInblock;

		dim3 thrs(threadsInblock);
		dim3 blcs(sresult);

		allocateGPU(&result, sresult*sizeof(float) );

		compare comp;
		checkError( cudaMemcpyFromSymbol( &comp, dev_compare, sizeof(compare) ) );
		compute_min_max<float> <<< blcs, thrs >>>(//gpu_array
												gpu_elements
												, comp
												, size
												, result);

		float* cpu_result = new float[sresult];
		copyFromDevice( &cpu_result[0], result, sresult * sizeof( float ) );
		float k = cpu_result[0];
		for(int i = 0; i < sresult; ++i)
		{
			if(cpu_compare(cpu_result[i], k) && cpu_result[i] < kInfinity)
				k = cpu_result[i];
		}

		delete[] cpu_result;
		//checkError( cudaFree(gpu_array) );
		checkError( cudaFree(result) );

	return k;
}
#endif
