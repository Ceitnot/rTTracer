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
#ifndef __ACCELERATION__
#define __ACCELERATION__

#include "geom.cuh"
/*!\file
 \brief Kay and Kajiya bounding volumes
 */
class Boundaries{
public:
	static const uint8_t nNormals = NORMALS;
	static const uint8_t nPlaneSet = NORMAL_SET;
	__host__ __device__
	Boundaries();
	/*!
	 * \brief hit test for a bounding volume
	 * \param [out] ray - sets tNear and tFar in order to determine the shortest distance on the ray
	 * \param[in] precomputed - numerator and denominator to accelerate the shorted hit distance
	 * \param[out] planeIndex - for displaying the bounding volume. That is debug needs.
	 * \return bool - if there was an intersection between a ray and the bounding volume
	 * */
	__host__ __device__
	bool hit(Ray &ray, float (&precomPuted)[2][nNormals], uint8_t *planeIndex = nullptr) const;
	/*!
	 * \brief returns the distance from the origin to the bounding volume
	 * \param [in] planePairNumber - a slab number to choose
	 * \param[in] farOrClose - choose a far or the closest bounding plane
	 * \return const float& - a const link to the distance from the origin to the bounding volume
	 * */
	__host__ __device__
	const float& at(size_t planePairNumber, size_t farOrClose) const;
	/*!
	 * \brief operator []
	 * \param[in] i - choose a far or the closest bounding plane
	 * \return float * - a pointer to distances
	 * */
	__host__ __device__
    float * operator[] (size_t i){
		return distances[i];
	}

private:
	//!each bounding volume can have its own number of bounding slabs
	  float distances[nNormals][nPlaneSet];
};

/*!
 *	\brief A priority queue that works on device.
 *	Collects pointers to the objects in the painter's algorithm manner.
 **/
template<class ObjectType, size_t number>
class __align__(16) DevPriorityQueue{
	ObjectType objects[number];
	int keys[number];
	size_t _size;

	__host__ __device__
	int parent(uint16_t i){
		if(i == 0) return i;
		return (i % 2) ? i/2 : i/2 - 1;
	}
	__host__ __device__
	int leftChild(uint16_t i){
		int child = 2*i + 1;
		return ( child < _size ) ? child : i;
	}
	__host__ __device__
	int rightChild(uint16_t i){
		int child = 2*i + 2;
		return ( child < _size ) ? child : i;
	}
	__host__ __device__
	void siftUp(){
		for(int i = _size - 1
				; i > 0 && keys[i] < keys[ parent(i) ]
				; i = parent(i)){
			algorithms::swap(keys[i], keys[ parent(i) ]);
			algorithms::swap(objects[i], objects[ parent(i) ]);
		}

	}
	__host__ __device__
	int minChildIndex(int num){
		return ( keys[ leftChild( num ) ] > keys[ rightChild( num ) ] )
				? rightChild( num ) : leftChild( num );
	}
	__host__ __device__
	void siftDown(){
		int childIndex = minChildIndex( 0 );
		for(int i = 0; keys[i] > keys[ childIndex ] ; ){

			algorithms::swap( keys[i], keys[ childIndex ] );
			algorithms::swap( objects[i], objects[ childIndex ] );
			i = childIndex;
			childIndex = minChildIndex( i );
		}
	}

public:
	__host__ __device__
	size_t size(){
		return _size;
	}
	__host__ __device__ DevPriorityQueue():_size(0){}
	__host__ __device__
	void push(int key, const ObjectType& object){
		assert(_size + 1 < number);
		keys[_size] = key;
		objects[_size] = object;
		++_size;
		siftUp();
	}
	__host__ __device__
	ObjectType pop(){
		assert(_size > 0);
		ObjectType object = objects[0];
		objects[0] = objects[_size - 1];
		keys[0] = keys[_size - 1];
		siftDown();
		--_size;
		return object;
	}
};
#endif
