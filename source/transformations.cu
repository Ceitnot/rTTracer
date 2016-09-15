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
#include "transformations.cuh"
#include <iostream>
SquareMatrix4f MeshTransformation::transform() const
{
		return mRotate * mScale * mTranslate;
}
SquareMatrix4f& MeshTransformation::rotate(const float3& _axis, float degAngle){
	float3 axis = normalize(_axis);
	    /* rotation quaternion
	     * Q = [Vx*sin(a/2), Vy*sin(a/2), Vz*sin(a/2), cos(a/2)] (1)
	     * or Q = [cos(a/2), U*sin(a/2)] (2) , where U - is normal vector axis, which
	     * represents the angle of rotation. So, 'a' increases - sin gets bigger
	      */
	const float SinHalfAngle = sinf(ToRadian(degAngle/2));
	    /*
	     * Half angle if half rotation in this method, so
	     * because of (2) this formula  W = Q*V*Q^(-1) initially rotates one half
	     * (let) Q' = Q*V (product is quaternian also)
	     * and finally it rotates the second half Q'*Q^(-1).
	     * We need two parts in order to rotate in hyper space and
	     * then to translate it in 3D space.
	     *  All because of the ordering of operations. */
	 const float CosHalfAngle = cosf( ToRadian(degAngle/2) );
	 float4 rotation;
	 rotation.x = axis.x * SinHalfAngle;
	 rotation.y = axis.y * SinHalfAngle;
	 rotation.z = axis.z * SinHalfAngle;
	 rotation.w = CosHalfAngle;
	 Quaternion RotationQ(rotation.x, rotation.y, rotation.z, rotation.w);
	 return mRotate = RotationQ.toMatrix();
	}
SquareMatrix4f& MeshTransformation::translate( float3 translationPoint){
	return	mTranslate = SquareMatrix4f( 1, 0, 0, 0,
								0, 1, 0, 0,
								0, 0, 1, 0,
								translationPoint.x,
								translationPoint.y,
								translationPoint.z, 1);
	}
SquareMatrix4f& MeshTransformation::scale( float3 scaleFactor ){
    return	mScale = SquareMatrix4f( scaleFactor.x, 0, 0, 0,
							0, scaleFactor.y, 0, 0,
							0, 0, scaleFactor.z, 0,
							0, 0, 0, 1);
	}
void MeshTransformation::reset(){
	mRotate = SquareMatrix4f::Identity;
	mTranslate = SquareMatrix4f::Identity;
	mScale = SquareMatrix4f::Identity;
}
MeshTransformation::~MeshTransformation(){}

	float3 MeshTransformation::normalize(float3 axis) const{
	float length = sqrtf(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
	float invLen = 1/length;
	return make_float3(axis.x*invLen, axis.y*invLen, axis.z*invLen);
	}
