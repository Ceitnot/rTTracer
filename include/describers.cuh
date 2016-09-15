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

#ifndef DESCRIBERS_CUH_
#define DESCRIBERS_CUH_

/*! \file
    \defgroup intersection_test_prperties Intersection tests and hit object properties
    \ingroup rendering
    @{
*/

/*!\fn __host__ __device__ bool intersectMesh( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
 	\brief	Checks if a ray intersects a mesh.
	\param[out] ray - the ray structure
	\param[out] obj wrapper with a specific object's data
	\param[out] ctxt structure gets texture coordinates etc.
*/
__host__ __device__
bool intersectMesh( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt);

/*! \fn __host__ __device__ bool intersectSphere( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
 	\brief	Checks if a ray intersects a sphere.
	\param[in] ray - the ray structure
	\param[out] obj - wrapper with a specific object's data
	\param[out] ctxt - structure gets texture coordinates etc.
*/
__host__ __device__
	 bool intersectSphere( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt);

/*! \fn __host__ __device__ bool intersectPlane( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
 	\brief	Checks if a ray intersects a plane.
	\param[in] ray - the ray structure
	\param[out] obj - wrapper with a specific object's data
	\param[out] ctxt - structure gets texture coordinates etc.
*/
__host__ __device__
	 bool intersectPlane( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt);
/*! \fn __host__ __device__ bool intersectCube( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
 	\brief	Checks if a ray intersects a cube.
	\param[in] ray - the ray structure
	\param[out] obj - wrapper with a specific object's data
	\param[out] ctxt - structure gets texture coordinates etc.
*/
__host__ __device__
	 bool intersectCube( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt);

/*! \fn __host__ __device__ bool intersectBoundingVolume(Ray& ray
							, float (&precompute)[2][7]
		                    , DevObject& currentObject
		                    , uint8_t& normalIndex)
	\brief	Checks if a ray intersects a bounding slabs of a triangle mesh.
	\param[in] ray - the ray structure
	\param[out] precompute - a precomputed( in ObjectBox::hit ) array of dot products divisions
	\param[out] currentObject wrapper with a specific object's data
	\param[out] normalIndex - needed to get a normal of a slab.
*/
__host__ __device__
bool intersectBoundingVolume(Ray& ray
							, float (&precompute)[2][7]
		                    , DevObject& currentObject
		                    , uint8_t& normalIndex);
/*! \fn __host__ __device__ void meshProperties (const DevObject& object,
												const Vec3f &hitPoint,
												GPU::RenderingContext& ctx,
												Vec2f &hitTextureCoordinates)
	\brief	Gets mesh properties on a hitpoint
	\param[in] object - wrapper with a specific object's data
	\param[out] hitPoint - output argument to get hit point
	\param[out] ctx - output argument to get texture barycentric coordinates
	\param[out] hitTextureCoordinates - texture coordinates with regards to barycentric coordinates of hit point.
*/
__host__ __device__
void meshProperties (const DevObject& object,
					const Vec3f &hitPoint,
					GPU::RenderingContext& ctx,
					Vec2f &hitTextureCoordinates);

/*! \fn __host__ __device__ void sphereProperties (const DevObject& object,
													const Vec3f &hitPoint,
													GPU::RenderingContext& ctx,
													Vec2f &hitTextureCoordinates)
	\brief	Gets sphere properties on a hitpoint.
	\param[in] object - wrapper with a specific object's data
	\param[out] hitPoint - output argument to get hit point
	\param[out] ctx - output argument to get texture barycentric coordinates
	\param[out] hitTextureCoordinates - texture coordinates with regards to barycentric coordinates of hit point.
*/
__host__ __device__
void sphereProperties (const DevObject& object,
						const Vec3f &hitPoint,
						GPU::RenderingContext& ctx,
						Vec2f &hitTextureCoordinates);
/*! \fn __host__ __device__ void planeProperties (const DevObject& object,
													const Vec3f &hitPoint,
													GPU::RenderingContext& ctx,
													Vec2f &hitTextureCoordinates)
	\brief	Gets plane properties on a hitpoint.
	\param[in] object - wrapper with a specific object's data
	\param[out] hitPoint - output argument to get hit point
	\param[out] ctx - output argument to get texture barycentric coordinates
	\param[out] hitTextureCoordinates - texture coordinates with regards to barycentric coordinates of hit point.
*/
__host__ __device__
void planeProperties (const DevObject& object,
						const Vec3f &hitPoint,
						GPU::RenderingContext& ctx,
						Vec2f &hitTextureCoordinates);
/*! \fn __host__ __device__ void cubeProperties (const DevObject& object,
												const Vec3f &hitPoint,
												GPU::RenderingContext& ctx,
												Vec2f &hitTextureCoordinates)
	\brief	Gets cube properties on a hitpoint.
	\param[in] object - wrapper with a specific object's data
	\param[out] hitPoint - output argument to get hit point
	\param[out] ctx - output argument to get texture barycentric coordinates
	\param[out] hitTextureCoordinates - texture coordinates with regards to barycentric coordinates of hit point.
*/
__host__ __device__
void cubeProperties (const DevObject& object,
					const Vec3f &hitPoint,
					GPU::RenderingContext& ctx,
					Vec2f &hitTextureCoordinates);
/*! \fn void deleteMesh(DevObject* object)
	\brief	Device memory deallocation function.
	\param[out] object - wrapper with a specific object's data
*/
void deleteMesh(DevObject* object);
/*! \fn __device__ void transformMesh(DevObject* object, const SquareMatrix4f& mTransform)
	\brief	Whole mesh translation, rotation and scale function.
	\param[out] object - wrapper with a specific object's data
	\param[in] mTransform - object-to-world matrix
*/
__device__
void transformMesh(DevObject* object, const SquareMatrix4f& mTransform);
/*! \fn __device__ void transformSphere(DevObject* object, const SquareMatrix4f& mTransform)
	\brief	Implicit sphere translation, rotation and scale function.
	\param[out] object - wrapper with a specific object's data
	\param[in] mTransform - object-to-world matrix
*/
__device__
void transformSphere(DevObject* object, const SquareMatrix4f& mTransform);
/*! \fn __device__ void transformPlane(DevObject* object, const SquareMatrix4f& mTransform)
	\brief	Implicit sphere translation, rotation and scale function.
	\param[out] object - wrapper with a specific object's data
	\param[in] mTransform - object-to-world matrix
*/
__device__
void transformPlane(DevObject* object, const SquareMatrix4f& mTransform);
/*! \fn __device__ void transformCube(DevObject* object, const SquareMatrix4f& mTransform)
	\brief	Implicit sphere translation, rotation and scale function.
	\param[out] object - wrapper with a specific object's data
	\param[in] mTransform - object-to-world matrix
*/
__device__
void transformCube(DevObject* object, const SquareMatrix4f& mTransform);
/*! \fn __device__ Vec3f  distantLightReflectedColor(const ILight* _light, GPU::RenderingContext& rctx, Vec3f& albedo)
	\brief	Appearence of incident light.
	\param[in] _light - wrapper with a specific light's data
	\param[in] rctx
	\param[in] albedo
*/
__device__
Vec3f  distantLightReflectedColor(const ILight* _light, GPU::RenderingContext& rctx, Vec3f& albedo);
/*! \fn __device__ void illuminateDistant(const  ILight& light, const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance)
	\brief	Implicit sphere translation, rotation and scale function.
	\param[in] light - wrapper with a specific light's data
	\param[in] P - not used here
	\param[in] lightDir - light direction.
	\param[in] lightIntensity
	\param[in] distance - not used here.
*/
__device__
void illuminateDistant(const  ILight& light, const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance);
/*! \fn __device__ void illuminatePoint(const ILight& light, const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance)
	\brief	Implicit sphere translation, rotation and scale function.
	\param [in] light - wrapper with a specific light's data
	\param [in] P - light source position.
	\param [out] lightDir - light direction.
	\param [out] lightIntensity
	\param [out] distance - distance to light source
*/
__device__
void illuminatePoint(const ILight& light, const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance);

/*!\fn  void setMeshAccelerationVolume(const DevObject* currentObject
									, Boundaries* boundaries
									, Vec3f (&boundingPlaneNormals)[7]
									,float * (&gpu_allDotProducts)[7] )
 * \brief The function gets bounding volume distances computed.
 * \param [in] currentObject - wrapper object with pointer to the stored in device memory specific instance.
 * \param [out] boundaries - to store result
 * \param [in] boundingPlaneNormals - bounding plane orientation normals.
 * \param [out] gpu_allDotProducts - allocated in advance device memory to store all possible distances the function uses.
 * */
void setMeshAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] );
/*!\fn void setSphereAccelerationVolume(const DevObject* currentObject
										, Boundaries* boundaries
										, Vec3f (&boundingPlaneNormals)[7]
										,float * (&gpu_allDotProducts)[7] )
 * \brief The function gets bounding volume distances computed.
 * \param [in] currentObject - wrapper object with pointer to the stored in device memory specific instance.
 * \param [out] boundaries - to store result
 * \warning Parameters down bellow are not used in this function:
 * \param boundingPlaneNormals
 * \param gpu_allDotProducts
 * */
void setSphereAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] );
/*!\fn void setPlaneAccelerationVolume(const DevObject* currentObject
									, Boundaries* boundaries
									, Vec3f (&boundingPlaneNormals)[7]
									,float * (&gpu_allDotProducts)[7] )
 * \brief The function gets bounding volume distances computed.
 * \param [in] object - wrapper object with pointer to the stored in device memory specific instance.
 * \param [out] boundaries - to store result
 * \param [in] boundingPlaneNormals - bounding plane orientation normals.
 * \param [out] gpu_allDotProducts - allocated in advance device memory to store all possible distances the function uses.
 * */
void setPlaneAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] );
/*!\fn void setCubeAccelerationVolume(const DevObject* currentObject
									, Boundaries* boundaries
									, Vec3f (&boundingPlaneNormals)[7]
									,float * (&gpu_allDotProducts)[7] )
 * \brief The function gets bounding volume distances computed.
 * \param [in] object - wrapper object with pointer to the stored in device memory specific instance.
 * \param [out] boundaries - to store result
 * \param [in] boundingPlaneNormals - bounding plane orientation normals.
 * \param [out] gpu_allDotProducts - allocated in advance device memory to store all possible distances the function uses.
 * */
void setCubeAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] );
/*!\fn  __device__ float strips(const Vec2f& hitTexCoordinates)
 * \brief Computes pattern at a particular point in texture.
 * \param [in] hitTexCoordinates
 * */
__device__ float strips (const Vec2f& hitTexCoordinates);
/*!\fn  __device__ float wave(const Vec2f& hitTexCoordinates)
 * \brief Computes pattern at a particular point in texture.
 * \param [in] hitTexCoordinates
 * \return zero or one
 * */
__device__ float wave (const Vec2f& hitTexCoordinates);
/*!\fn  __device__ float grid(const Vec2f& hitTexCoordinates)
 * \brief Computes pattern at a particular point in texture.
 * \param [in] hitTexCoordinates
 * \return a normalized floating point number
 * */
__device__ float grid (const Vec2f& hitTexCoordinates);
/*!\fn  __device__ float checker(const Vec2f& hitTexCoordinates)
 * \brief Computes pattern at a particular point in texture.
 * \param [in] hitTexCoordinates
 * \return a normalized floating point number
 * */
__device__ float checker (const Vec2f& hitTexCoordinates);
/*!\fn  __device__ float none(const Vec2f& hitTexCoordinates)
 * \brief Function just for capability with pattern calculation.
 * \param [in] hitTexCoordinates
 * \return zero or one
 * */
__device__ float none(const Vec2f& hitTexCoordinates);

/*!
    @}
*/

/*! \fn Mesh* loadMesh(std::string filename, const SquareMatrix4f & model)
	\brief	Function to load and transform mesh.
	\param [in] filename - file with a mesh
	\param [in] model - object-to-world matrix
*/
Mesh* loadMesh(std::string filename, const SquareMatrix4f & model);
#endif /* DESCRIBERS_CUH_ */
