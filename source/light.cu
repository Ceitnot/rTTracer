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
#include "light.cuh"

	ILight::ILight(const SquareMatrix4f &ltoW
			, const Vec3f &color
			, const float &i
			, uint8_t lightType)
								:
								_data(nullptr)
								,lightType(lightType)
								,illuminate(nullptr)
								, reflectedColor(nullptr)
								, color(color)
								, model(ltoW)
								, intensity(i)
								{

	}
	__host__ __device__
	ILight::ILight(const ILight& light):
									_data(light._data)
									, lightType(light.lightType)
									, illuminate(light.illuminate)
									, reflectedColor(light.reflectedColor)
									, color(light.color)
									, intensity(light.intensity)
									, model(light.model)
									{}
	ILight::~ILight(){
		cudaFree(_data);
	}

	__host__ __device__
	ILight& ILight::operator=(const ILight& light){
		ILight tmp(light);
		algorithms::swap(*this, tmp);
		return *this;
	}
    __host__ __device__
    const void* ILight::data() const{
    	return _data;
    }
    __host__ __device__
    void* ILight::data(){
    	 return const_cast<void *>(
    			 static_cast<const ILight*>(this)->data() );
    }
