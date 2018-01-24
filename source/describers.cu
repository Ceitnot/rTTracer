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
#include"objectBox.cuh"
#include "rendering.cuh"
#include "global_constants.cuh"
void deleteMesh(DevObject* object)
{
	checkError( cudaFree( const_cast<void *>( object->data() ) ) );
}
__host__ __device__
 bool intersectMesh( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
{
	const Mesh * mesh = reinterpret_cast<const Mesh *> (obj.data());

	bool isect = false;
	Vec3f triVerts[3];
	float t = kInfinity;
	for (uint32_t i = 0; i < mesh->numTris; ++i) {

		triVerts[0] =  mesh->P[mesh->triangles[i][0]];
		triVerts[1] =  mesh->P[mesh->triangles[i][1]];
		triVerts[2] =  mesh->P[mesh->triangles[i][2]];
		float u = 0, v = 0;

		if ( mesh->triangles->hit(ray, triVerts, u, v) && ray.tNearest < t){
		  //save barycentric coordinates
			ctxt.textureCoordinates.xy.x = u;
			ctxt.textureCoordinates.xy.y = v;
			ctxt.triangleIndex = i;// сохраняем индекс пересеченного трейгольника.
			t = ray.tNearest;
			isect = true;
		}
	}
	ray.tNearest = t;
	return isect;
}
//intersect sphere

__host__ __device__
bool intersectSphere( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
{
	 const Sphere * sphere = reinterpret_cast<const Sphere *> (obj.data());
	 volatile const Sphere _sphere( *sphere );
	float t0, t1; //two distances
	// hypotenuse
	Vec3f L = sphere->center() - ray.orig;
	//find a projection length from origin to the sphere center
	//on the ray
	float tca = L.dot(ray.dir);
	//an intersection can happen only in front of the camera
	if ( tca < 0 ) return false;
	//d2 - a cathetus length of the right triangle from a sphere center to the ray
	//use Pythagorian theorem. L.L - squared L length

	float d2 = L.dot(L) - tca * tca;
	//if d2 > than squared radius
	//then ray definitely misses the sphere
	if ( d2 > sphere->squareRadius() ) return false;
	//use pathegorian theorem again to find
	// the length between d and hit point

	float thc = sqrt( sphere->squareRadius() - d2);

	// find nearest and farthest distance on the ray
	t0 = tca - thc;
	t1 = tca + thc;

	if (t0 > t1) algorithms::swap(t0, t1);

	if (t0 < 0) {
		t0 = t1; // if t0 is negative, let's use t1 instead
		if (t0 < 0) return false; // both t0 and t1 are negative
	}

	ray.t = t0;

   if( ray.t > ray.tMax || ray.t < ray.tMin) return false;
	//save positive nearest point on the ray if calculated value is less then previous one
	//or return false and don't save any
  return  ray.t < ray.tNearest ? ray.tNearest = ray.t : false;

}
__host__ __device__
bool planeTest(const Plane &plane, Ray& ray)
{
    float denom = plane.normal().dot(ray.dir.normalize());
    if (denom > 1e-6) {
        Vec3f p0l0 = plane[0] - ray.orig;
        ray.tNearest = p0l0.dot(plane.normal()) / denom;
        return (ray.tNearest >= 0);
    }

    return false;
}
__host__ __device__
bool intersectPlane( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt){
	const Plane &plane = *reinterpret_cast<const Plane *> (obj.data());
	return planeTest(plane, ray);
}
__host__ __device__
bool intersectCube( Ray& ray, DevObject& obj, GPU::RenderingContext& ctxt)
{
	const Cube &cube = *reinterpret_cast<const Cube *> (obj.data());
	uint8_t indexY = 3, indexZ = 0; //to store plane indices
	ctxt.index = 4;
	float tmin = (cube.min().xyz.x - ray.orig.xyz.x) / ray.dir.xyz.x;
	float tmax = (cube.max().xyz.x - ray.orig.xyz.x) / ray.dir.xyz.x;

	if (tmin > tmax) {
		algorithms::swap(tmin, tmax);
		ctxt.index = 5;
	}

	float tymin = (cube.min().xyz.y - ray.orig.xyz.y) / ray.dir.xyz.y;
	float tymax = (cube.max().xyz.y - ray.orig.xyz.y) / ray.dir.xyz.y;

	if (tymin > tymax) {
		algorithms::swap(tymin, tymax);
		indexY = 3;
	}

	if ((tmin > tymax) || (tymin > tmax))
	   return false;

	if (tymin > tmin){
	   tmin = tymin;
	   ctxt.index = indexY;
	}

	if (tymax < tmax){
	    tmax = tymax;
	}

	float tzmin = (cube.min().xyz.z - ray.orig.xyz.z) / ray.dir.xyz.z;
	float tzmax = (cube.max().xyz.z - ray.orig.xyz.z) / ray.dir.xyz.z;

	if (tzmin > tzmax) {
		algorithms::swap(tzmin, tzmax);
		indexZ = 2;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
	   return false;

	if (tzmin > tmin){
	   tmin = tzmin;
	   ctxt.index = indexZ;
	}

	if (tzmax < tmax)
	   tmax = tzmax;

	ray.t = tmin;

	if( ray.t > ray.tMax || ray.t < ray.tMin) return false;
	//save positive nearest point on the ray if calculated value is less then previous one
	//or return false and don't save any
	return ray.t < ray.tNearest ? ray.tNearest = ray.t : false;
}
__host__ __device__
bool intersectBoundingVolume(Ray& ray, float (&precompute)[2][7], DevObject& currentObject, uint8_t& normalIndex)
{

	const Boundaries * bv = reinterpret_cast<const Boundaries *>( currentObject.boundaries() );
	return bv->hit(ray, precompute, &normalIndex);
}
//mesh
__host__ __device__
void meshProperties (const DevObject& object,
	        const Vec3f &hitPoint,
	        GPU::RenderingContext& ctx,
	        Vec2f &hitTextureCoordinates) {
	const Mesh * mesh = reinterpret_cast<const Mesh *> (object.data());
	         // face normal
	/*
	const Vec3f &v0 = mesh->P[ mesh->triangles[ctx.triangleIndex][0] ];
	const Vec3f &v1 = mesh->P[ mesh->triangles[ctx.triangleIndex][1] ];
	const Vec3f &v2 = mesh->P[ mesh->triangles[ctx.triangleIndex][2] ];
	ctx.hitNormal = (v1 - v0).cross(v2 - v0);
	ctx.hitNormal.normalize();
*/
	// vertex normal

	const Vec3f &n0 = mesh->N[ mesh->triangles[ctx.triangleIndex][0] ];
	const Vec3f &n1 = mesh->N[ mesh->triangles[ctx.triangleIndex][1] ];
	const Vec3f &n2 = mesh->N[ mesh->triangles[ctx.triangleIndex][2] ];
	ctx.hitNormal = (1 - ctx.textureCoordinates.xy.x - ctx.textureCoordinates.xy.y)
			* n0 + ctx.textureCoordinates.xy.x * n1 + ctx.textureCoordinates.xy.y * n2;
	ctx.hitNormal.normalize();

	const Vec2f &st0 = mesh->texCoordinates[ctx.triangleIndex * 3];
	const Vec2f &st1 = mesh->texCoordinates[ctx.triangleIndex * 3 + 1];
	const Vec2f &st2 = mesh->texCoordinates[ctx.triangleIndex * 3 + 2];

	hitTextureCoordinates = (1 - ctx.textureCoordinates.xy.x
							- ctx.textureCoordinates.xy.y)* st0
							+ ctx.textureCoordinates.xy.x * st1
					+ ctx.textureCoordinates.xy.y * st2;
}
//sphere
__host__ __device__
void sphereProperties (const DevObject& object,
	        const Vec3f &hitPoint,
	        GPU::RenderingContext& ctx,
	        Vec2f &hitTextureCoordinates)
{
	const Sphere * sphere = reinterpret_cast<const Sphere *> (object.data());
	ctx.hitNormal = hitPoint - sphere->center();
	ctx.hitNormal.normalize();
    //compute cpherical coordinates in order to find texture coordinates
    hitTextureCoordinates.xy.x = (1 + atan2(ctx.hitNormal.xyz.z, ctx.hitNormal.xyz.x) / M_PI) * 0.5;
    hitTextureCoordinates.xy.y = acosf(ctx.hitNormal.xyz.y) / M_PI;
}
__host__ __device__
	void getPlaneProperties(const Plane &plane
							,const Vec3f &hitPoint
							,GPU::RenderingContext& ctx
							,Vec2f &hitTextureCoordinates)
{
	ctx.hitNormal = -plane.normal();
	float2 uv = texCoordinates(plane, hitPoint);
	hitTextureCoordinates.xy.x = uv.x/plane.width();
	hitTextureCoordinates.xy.y = uv.y/plane.height();
}
__host__ __device__
void planeProperties (const DevObject& object,
	        const Vec3f &hitPoint,
	        GPU::RenderingContext& ctx,
	        Vec2f &hitTextureCoordinates)
{
	const Plane &plane = *reinterpret_cast<const Plane *> (object.data());
	getPlaneProperties(plane, hitPoint, ctx, hitTextureCoordinates);
}
__host__ __device__
void cubeProperties (const DevObject& object,
	        const Vec3f &hitPoint,
	        GPU::RenderingContext& ctx,
	        Vec2f &hitTextureCoordinates)
{
	const Cube &cube = *reinterpret_cast<const Cube *> (object.data());
	ctx.hitNormal = cube.normal(ctx.index);
	Plane plane(cube[  cube.vIndex(ctx.index, 0)], cube[ cube.vIndex(ctx.index, 1)], cube[ cube.vIndex(ctx.index, 2)],
			ctx.hitNormal);
	float2 uv = texCoordinates(plane, hitPoint);
	hitTextureCoordinates.xy.x = uv.x/plane.width();
	hitTextureCoordinates.xy.y = uv.y/plane.height();
}
//mesh
 __device__
void transformMesh(DevObject* object, const SquareMatrix4f& mTransform)
{
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	Mesh& mesh = *reinterpret_cast<Mesh *>( object->data() );

	__shared__ Vec3f vertexBuffer[THREADS_ALONG_X];
	vertexBuffer[threadIdx.x] = mesh[offset];
	mTransform.mulPointMat( vertexBuffer[threadIdx.x]
	                                      , vertexBuffer[threadIdx.x] );
	mesh[offset] = vertexBuffer[threadIdx.x];
}
 __device__
void transformSphere(DevObject* object, const SquareMatrix4f& mTransform)
{
	Sphere& sphere = *reinterpret_cast<Sphere *>( object->data() );
	//no scale yet
	mTransform.mulPointMat( sphere.center(), sphere.center() );

}
 __device__
 void computePlane(Plane& plane, const SquareMatrix4f& mTransform)
 {
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ Vec3f vertexBuffer[THREADS_ALONG_X];
	vertexBuffer[threadIdx.x] = plane[offset];
	mTransform.mulPointMat( vertexBuffer[threadIdx.x]
										  , vertexBuffer[threadIdx.x] );
	plane[offset] = vertexBuffer[threadIdx.x];
 }
 __device__
 void transformPlane(DevObject* object, const SquareMatrix4f& mTransform)
 {
	Plane& plane = *reinterpret_cast<Plane *>( object->data() );
	computePlane(plane, mTransform);
 }

 __device__
 void transformCube(DevObject* object, const SquareMatrix4f& mTransform)
 {
	Cube &cube = *reinterpret_cast< Cube *> (object->data());
	for(int i = 0; i < Cube::vertsNumber; ++i)
	{
		mTransform.mulPointMat( cube[i], cube[i] );
	}
 }

 __device__
 void illuminateDistant(  const ILight& light
		 	 	 	 	 , const Vec3f &P
						 , Vec3f &lightDir
						 , Vec3f &lightIntensity
						 , float &distance)
 {
	 const DistantLight<Vec3f>& dlight = *reinterpret_cast<const DistantLight<Vec3f> *>( light.data() );
     lightDir = dlight.dir;
     lightIntensity = light.color * light.intensity;
     distance = kInfinity;
 }
 __device__
 void illuminatePoint(   const ILight& light
		 	 	 	 	 , const Vec3f &P
						 , Vec3f &lightDir
						 , Vec3f &lightIntensity
						 , float &distance)
 {
	 const PointLight<Vec3f>& plight = *reinterpret_cast<const PointLight<Vec3f> *>( light.data() );
     lightDir = (P - plight.pos);
     distance = lightDir.length();
     float r2 = distance*distance;
     lightDir.xyz.x /= distance, lightDir.xyz.y /= distance, lightDir.xyz.z /= distance;
     // avoid division by 0
     lightIntensity = light.color * light.intensity / (4 * M_PI * r2);
 }
//light

__device__
Vec3f  distantLightReflectedColor(const ILight* _light, GPU::RenderingContext& rctx, Vec3f& albedo)
{

	const DistantLight<Vec3f> &light = *reinterpret_cast< const DistantLight<Vec3f> *>( _light->data() );
	return (albedo / M_PI
			* _light->intensity
			*_light->color
			* fmaxf (0.f, rctx.hitNormal.dot(-light.dir) ));
}
/*! \file
 * \fn template <class ObjectType> void setAccelerationVolume(const ObjectType* object, Boundaries* boundaries, Vec3f (&boundingPlaneNormals)[7], float * (&gpu_allDotProducts)[7] )
 * \brief The wrapper function gets bounding volume distances computed.
 * \param [in] object - wrapper object with pointer to the stored in device memory specific instance.
 * \param [out] boundaries - to store result
 * \param [in] boundingPlaneNormals - bounding plane orientation normals.
 * \param [out] gpu_allDotProducts - allocated in advance device memory to store all possible distances the function uses.
 * \bug thrust library allocation causes a bad_alloc
 * */
template <class ObjectType>
void setAccelerationVolume(const ObjectType* object
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] )
{
	thrust::device_vector<float> nearFar(
	Boundaries::nNormals * Boundaries::nPlaneSet );
		//compute min and max dot product of each normal with every vertex
		cudaStream_t streams[Boundaries::nNormals];

	#pragma omp parallel for
		for(int i = 0; i < Boundaries::nNormals; ++i)
		{
			cudaStreamCreate(&streams[i]);

			//compute all dot products of bounding planes normals and all of object point
			GPU::massiveDotProduct<ObjectType>
				<<< gridDIM( object->vertsNumber, THREADS_ALONG_X)
				, THREADS_ALONG_X, 0,streams[i] >>>
				 ( object
				, boundingPlaneNormals[i]
				, gpu_allDotProducts[i]
				, object->vertsNumber);

			//synchronize GPU threads and CPU threads
			cudaDeviceSynchronize();

			thrust::device_vector<float> vectorDots( gpu_allDotProducts[i]
					, gpu_allDotProducts[i] + object->vertsNumber) ;
			thrust::device_vector<float>::iterator iterMin =thrust::min_element(
					vectorDots.begin()
					, vectorDots.end() );
			thrust::device_vector<float>::iterator iterMax = thrust::max_element( vectorDots.begin()
					, vectorDots.end() );

			nearFar[ i * 2 ] = *iterMin;
			nearFar[ i * 2 + 1] = *iterMax;

		}
	 cudaDeviceSynchronize();
	 GPU::assignBoundingObject < Boundaries, float *, Boundaries::nNormals  >
			<<< gridDIM(Boundaries::nNormals, THREADS_ALONG_X)
			, THREADS_ALONG_X >>>
			( boundaries, thrust::raw_pointer_cast( nearFar.data() ));

}
void setPlaneAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] )
{
	const Plane * plane = reinterpret_cast<const Plane *> (currentObject->data());
	setAccelerationVolume<Plane>(plane, boundaries, boundingPlaneNormals, gpu_allDotProducts);
}

void setCubeAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] )
{
	const Cube * cube = reinterpret_cast<const Cube *> (currentObject->data());
	setAccelerationVolume<Cube>(cube, boundaries, boundingPlaneNormals, gpu_allDotProducts);
}

void setMeshAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] )
{
	const Mesh * mesh = reinterpret_cast<const Mesh *> (currentObject->data());
	setAccelerationVolume<Mesh>(mesh, boundaries, boundingPlaneNormals, gpu_allDotProducts);
}
//sphere
void setSphereAccelerationVolume(const DevObject* currentObject
								, Boundaries* boundaries
								, Vec3f (&boundingPlaneNormals)[7]
								,float * (&gpu_allDotProducts)[7] )
{
	const Sphere * sphere = reinterpret_cast<const Sphere *> (currentObject->data());

	float * points;
	size_t pointsSize = Boundaries::nNormals * Boundaries::nPlaneSet * sizeof(float);
	allocateGPU(&points,  pointsSize);

	GPU::getPoints<Sphere><<<Boundaries::nNormals, Boundaries::nPlaneSet>>>( sphere, points, pointsSize);

	GPU::assignBoundingObject < Boundaries, float *, Boundaries::nNormals  >
			<<< gridDIM(Boundaries::nNormals, THREADS_ALONG_X)
			, THREADS_ALONG_X >>>
			( boundaries, points);

}

__device__
float strips (const Vec2f& hitTexCoordinates)
{
	float angle = deg2rad(45);
    float s = hitTexCoordinates.xy.x * cosf(angle) - hitTexCoordinates.xy.y * sinf(angle);
    float scaleS = 20;
    return (modulo(s * scaleS) < 0.5);
}
__device__
float wave (const Vec2f& hitTexCoordinates)
{
	float scaleS = 10; // scale of the pattern
	return (sinf(hitTexCoordinates.xy.x * 2 * M_PI * scaleS) + 1) * 0.5;
}
__device__
float grid (const Vec2f& hitTexCoordinates)
{
	float scaleS = 10, scaleT = 10; // scale of the pattern
	return (cos(hitTexCoordinates.xy.y * 2 * M_PI * scaleT) * sin(hitTexCoordinates.xy.x * 2 * M_PI * scaleS) + 1) * 0.5;
}
__device__
float checker (const Vec2f& hitTexCoordinates)
{
	float angle = deg2rad(45);
    float s = hitTexCoordinates.xy.x * cosf(angle) - hitTexCoordinates.xy.y * sinf(angle);
    float t = hitTexCoordinates.xy.y * cosf(angle) + hitTexCoordinates.xy.x * sinf(angle);
    float scaleS = 5, scaleT = 5;
    return (modulo(s * scaleS) < 0.5) ^ (modulo(t * scaleT) < 0.5);
}
__device__
float none (const Vec2f& hitTexCoordinates)
{
	return 1;
}
