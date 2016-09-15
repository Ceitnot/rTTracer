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
#ifndef LIGHT_CUH_
#define LIGHT_CUH_
#include "geom.cuh"
#include "helper.cuh"

/*!
\defgroup lights Lights
 \ingroup wrapping_and_description
@{
 */

enum lights{
	DISTANT_LIGHT, POINT_LIGHT
};

class ILight;

typedef void (*Illuminate) (const ILight&
							, const Vec3f &
							, Vec3f &
							, Vec3f &
							, float &);
typedef Vec3f (*ReflectedColor)(const ILight*
		, GPU::RenderingContext&
		, Vec3f& );


//delta lights:
//unaffected by scale

/*!
Distant lights (sun light for example) -- emitted rays are parallel to each other.@n
	-Requires only a direction.@n
	-Affected by rotation@n
	-Unaffected by translation.@n
	*/
template <typename DirectionType>
class DistantLight
{
public:
	 __host__ __device__
    DistantLight(){
	 }
	 __host__ __device__
    DistantLight(const SquareMatrix4f & model){
		 model.mulVecMat(Vec3f(0, 0, -1)
				 	 	 	 , dir);
		 dir.normalize();
	 }
	 DirectionType dir;
};


/*!
 * Spherical light
 * \brief requires distance, unaffected by rotation
 */
template <typename PointType>
class PointLight
{
public:
	 __host__ __device__
    PointLight(){
	 }
	 __host__ __device__
	 PointLight(const SquareMatrix4f & model){
		 model.mulPointMat(Vec3f(0), pos);
	 }
	 PointType pos;
};

/*!
 \brief Area lights
 */
class ILight{
	void* _data;

public:
	const uint8_t lightType;
	__host__ __device__
	ILight(const SquareMatrix4f &l2w
			, const Vec3f &color
			, const float &i
			, uint8_t lightType);
	__host__ __device__
	ILight(const ILight& light);
	__host__ __device__
	~ILight();
template<class LightType>
	void init(ReflectedColor& rc,
			Illuminate& illum){
		devAllocateLight<LightType>();
		try{
		 //init light processing
			 checkError(
					 cudaMemcpyFromSymbol(
							 &reflectedColor
							 , rc
							 , sizeof(ReflectedColor) ) );
			 checkError(
					 cudaMemcpyFromSymbol(
							 &illuminate
							 , illum
							 , sizeof(Illuminate) ) );
		}catch( const std::exception & e){
			throw CudaException(" ILight::init -> ") + e;
		}
	}
	template<class LightType>
	void devAllocateLight(){
		//init light
		LightType tmp(model);
		try{
		 allocateGPU( &_data , sizeof(LightType));
		 copyToDevice(
		    		reinterpret_cast< LightType *>(_data)
		    										, &tmp
		    										, sizeof( LightType )
		    										);
		}catch( const std::exception & e){
			throw CudaException(" ILight::devAllocateLight -> ") + e;
		}
	}
	__host__ __device__
	ILight& operator=(const ILight& light);
    __host__ __device__
    const void* data() const;
    __host__ __device__
    void* data();

	Illuminate illuminate;
	ReflectedColor reflectedColor;

    Vec3f color;
    float intensity;
    SquareMatrix4f model;
};

/*!
@}
 */
#endif /* LIGHT_CUH_ */
