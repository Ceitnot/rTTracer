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
#ifndef RENDERING_CUH_
#define RENDERING_CUH_
#include "transformations.cuh"
#include "thrustHelper.cuh"
#include <omp.h>
/*
__host__ __device__
inline
float deg2rad(const float &deg) { return deg * M_PI / 180; }
*/

namespace GPU{
/*!\file
 \brief Rendering. Backward ray tracing.
 \addtogroup rendering Rendering
    @{
 */
	void* launchAsynchRayTracing(Camera cam
			, ObjectBox& objects
			, unsigned char * dev_bitmap
			, dim3& blocksPerGrid
			, dim3& threadsPerBlock);
/*!
\fn void render( Camera& cam,  ObjectBox& objects, unsigned char* buffer)
\brief A device kernel launcher
\callgraph
*/
	void render( Camera& cam,  ObjectBox& objects, unsigned char* buffer);

	__device__
	bool trace(
		 Ray& ray,
		 ObjectBox& objects,
		 RenderingContext& rctx
		, DevObject **hitObject
		, RayType raytype = PRIMARY_RAY);

	__device__
	Vec3f shade(
		 Ray &ray,
		 ObjectBox& obj,
		const Camera &options
		);
	/*!
	    @}
	 */
template <class BoundingVolumeType, class Distances, size_t size>
__global__ void assignBoundingObject( BoundingVolumeType *bv,  Distances nearFarDistances, size_t _size = size){
	BoundingVolumeType &_bv = *bv;
	if(threadIdx.x < size){
		_bv[threadIdx.x][0] = nearFarDistances[ threadIdx.x * 2 ];
		_bv[threadIdx.x][1] = nearFarDistances[threadIdx.x * 2 + 1 ];
	}
}
template <class ObjectType>
__global__ void getPoints( const ObjectType* object, float * points, size_t num){
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < num){
		points[tid] = (*object)[tid];
	}
}

template<class ObjectType>
__global__
void massiveDotProduct(const ObjectType* object
		, const Vec3f _boundingNormal
		, float* _res
		, size_t resLen){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const ObjectType& _object = *object;
	__shared__ float  pDots[ THREADS_ALONG_X ];
	if( tid < resLen)
	{
		pDots[ threadIdx.x ]
		            = _boundingNormal.dot(_object[tid]);
		__syncthreads();
	}
	_res[tid] = pDots[threadIdx.x];

 }
/*\fn bool isPointVisible(ObjectBox& obj, const Vec3f& hitPoint, const RenderingContext& rctx)
 *\brief a shading computation device function
 *\param[in] - obj - to get light and object to cast shadow on
 *\param[in] - hitPoint - ray-object intersection point
 *\param[in] - rctx - to get normal at the intersection point
 *\param[in] - bias - to avoid shadow acne
 *\return  - is a point in a shadow
 * */
__device__
bool isPointVisible(ObjectBox& obj
		, const uint8_t& lightNumber
		, const Vec3f& hitPoint
		, RenderingContext& rctx
		, const float& bias = 0.00001f);

}
//initial function
/*!
 \addtogroup rendering Rendering
    @{
 */
__global__ void raytracer( Camera cam
		, ObjectBox objects
	, unsigned char * dev_bitmap );
/*!
   @}*/
#endif /* RENDERING_CUH_ */
