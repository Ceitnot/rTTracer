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
#ifndef TRANSFORMATIONS_CUH_
#define TRANSFORMATIONS_CUH_
#include "objectBox.cuh"
/*!
 * \brief Interface
 * \detail Convenient way to create and store a matrix transformation pipeline.@n
 * \detail a host-side implementation.@n
 * \detail
 * */
class Pipeline{
public:
	virtual float3 normalize(float3 axis) const = 0;
	virtual SquareMatrix4f  transform() const = 0;
	virtual SquareMatrix4f& rotate(const float3& axis, float degAngle)  = 0;
	virtual SquareMatrix4f& translate( float3 translationPoint)  = 0;
	virtual SquareMatrix4f& scale( float3 scaleFactor )  = 0;
	virtual void reset() = 0;
	virtual ~Pipeline(){}
};
class MeshTransformation : public Pipeline{
	SquareMatrix4f mRotate;
	SquareMatrix4f mTranslate;
	SquareMatrix4f mScale;
public:
	/*!
	 * Axis normalization. Required for rotation.
	 * */
	float3 normalize(float3 axis) const;
	/*!
	 * \brief The function is called after applying all the transformations.
	 * */
	SquareMatrix4f transform() const;
	/*!
	 * \brief Rotation was implemented with Quaternions.
	 * */
	SquareMatrix4f& rotate(const float3& axis, float degAngle);
	SquareMatrix4f& translate( float3 translationPoint);
	SquareMatrix4f& scale( float3 scaleFactor );
	void reset();
	~MeshTransformation();
};
#endif /* TRANSFORMATIONS_CUH_ */
