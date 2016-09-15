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
#include "rendering.cuh"
#include "bitmap.cuh"
#include "describers.cuh"
void* GPU::launchAsynchRayTracing(Camera cam
		, ObjectBox& objects
		, unsigned char * dev_bitmap
		, dim3& blocksPerGrid
		, dim3& threadsPerBlock){
	raytracer<<<blocksPerGrid, threadsPerBlock>>>( cam , objects, dev_bitmap );
	return nullptr;
}

void GPU::render( Camera& cam,  ObjectBox& objects, unsigned char* buffer){

#ifdef __ASYNCH
	dim3 grid_dim(gridDIM(cam.partialWidth(), THREADS_ALONG_X), gridDIM(cam.partialHeight(), THREADS_ALONG_X ) );
    dim3 block_dim(THREADS_ALONG_X, THREADS_ALONG_X);
#ifdef __OMP
	cudaStream_t streams[Boundaries::nNormals];

#pragma omp parallel for
    for(int i = 0; i < DIVISION_FACTOR; ++i){
    	for ( int j = 0; j < DIVISION_FACTOR; ++j){
    		cudaStreamCreate(&streams[i]);
    		cam.offset( j*cam.partialWidth(), i*cam.partialHeight() );
    		raytracer<<<grid_dim, block_dim, 0, streams[i]>>>( cam , objects, buffer );
    	}
    }
#else
    std::thread cpuThreads[DIVISION_FACTOR][DIVISION_FACTOR];
    for(int i = 0; i < DIVISION_FACTOR; ++i){
    	for ( int j = 0; j < DIVISION_FACTOR; ++j){
    		cam.xOffset_ = j*cam.partialWidth();
    		cam.yOffset = i*cam.partialHeight();
    		cpuThreads[j][i] = std::thread(GPU::launchAsynchRayTracing, cam
    				, std::ref(objects)
    			, dev_bitmap
    			, std::ref(grid_dim)
    			, std::ref(block_dim) );
    	}
    }
    for(int i = 0; i < DIVISION_FACTOR; ++i){
    	for ( int j = 0; j < DIVISION_FACTOR; ++j){
    		cpuThreads[j][i].join();
    	}
    }
#endif

#else
	dim3 grid_dim(gridDIM(WIDTH, THREADS_ALONG_X), gridDIM(HEIGHT, THREADS_ALONG_X) );
    dim3 block_dim(THREADS_ALONG_X, THREADS_ALONG_X );
    //kernel call is asynchronous

    raytracer<<<grid_dim, block_dim>>>( cam , objects, buffer );

#endif
}
__device__
bool GPU::trace(
     Ray& ray,
     ObjectBox& objects,
     RenderingContext& rctx
    , DevObject **hitObject
    , RayType raytype){
    //for actual rendering of an object not a bv
	float t = kInfinity;; // one object might overlap the others
    DevPriorityQueue<DevObject *, MAX_OBJECTS> queue;
    #ifdef __BOUNDING
     if( objects.hit(ray, queue, rctx) ){
	   	ray.tNearest = kInfinity;

	   	RenderingContext current;

	   	while(queue.size() && *hitObject == nullptr){
	   		 *hitObject = queue.pop();
	   		 bool hit = false;
			 if(!(hit = (*hitObject)->intersect(ray, **hitObject, rctx) ) ){
				 //to reset bounding value to be visible
				*hitObject = nullptr;
			 }else if(hit && t > ray.tNearest){
				 t = ray.tNearest;
			 }
	   	}

   }
#else
    for (uint32_t k = 0; k < objects._size; ++k) {

    	RenderingContext current;

        if (objects[k].intersect(ray, objects[k], current) &&  ray.tNearest < tNearTriangle )
        	{
        		*hitObject = &objects[k];
        		tNearTriangle = ray.t;
        		rctx = current;
        	}
    }


#endif
    ray.tNearest = t;
    return (*hitObject != nullptr);
}
__device__
bool GPU::isPointVisible(ObjectBox& obj
						, const uint8_t& lightNumber
						, const Vec3f& hitPoint
						, RenderingContext& rctx
						, const float& bias){
	 RenderingContext shadingCtx;
	 DevObject *hitObject = nullptr;
	 float shadeIntersect;

	 obj.lights[lightNumber].illuminate(obj.lights[lightNumber]
									, hitPoint
									, rctx.lightDir
									, rctx.lightIntensity
									, shadeIntersect);

	 Ray shadowRay(hitPoint + rctx.hitNormal * bias, -rctx.lightDir);
	 return !trace(shadowRay, obj, shadingCtx, &hitObject, SHADOW_RAY);
}
__device__
Vec3f colorify( const float* _pattern, const Vec3f (&color)[2] ){
	const float &pattern = *_pattern;
	if(_pattern){
		return (pattern) ? color[1]*pattern : color[0];
	}else{
		return  color[0];
	}
}
__device__ void pattern(const DevObject* hitObject, const Vec2f& hitTexCoordinates, float& _pattern, float** pattern_ptr){
	 if(hitObject->appearence.texture){
		 _pattern = hitObject->pattern(hitTexCoordinates);
		 *pattern_ptr = &_pattern;
	 }
}
__device__
Vec3f reflect(const Vec3f &incident, const Vec3f &normal)
{
	return incident - 2 * incident.dot(normal) * normal;
}
__device__
void newCoordinateSystem(const Vec3f &N, Vec3f &Nt, Vec3f &Nb)
{
    if (fabs(N.xyz.x) > fabs(N.xyz.y))
        Nt = Vec3f(N.xyz.z, 0, -N.xyz.x) / sqrtf(N.xyz.x * N.xyz.x + N.xyz.z * N.xyz.z);
    else
        Nt = Vec3f(0, -N.xyz.z, N.xyz.y) / sqrtf(N.xyz.y * N.xyz.y + N.xyz.z * N.xyz.z);
    Nb = N.cross(Nt);
}

__device__
Vec3f _diffuse(ObjectBox& obj
			, DevObject *hitObject
			, const Vec3f& hitPoint
			, const Vec2f& hitTexCoordinates
			,GPU::RenderingContext& rctx){
	Vec3f hitColor;
	 for (uint32_t i = 0; i < obj.lightsNumber; ++i){
		 bool visible = isPointVisible(obj, i, hitPoint, rctx, rctx.bias);
	     float *pattern_ptr = nullptr;
	     float _pattern = 0, shadow;
		 pattern(hitObject, hitTexCoordinates, _pattern, &pattern_ptr);
		 shadow = (visible) ? 1 : 0.7;
		 hitColor += shadow * rctx.lightIntensity * colorify(pattern_ptr, hitObject->appearence.colors ) * fmaxf( 0.f, rctx.hitNormal.dot(-rctx.lightDir) );
	 }
	 return hitColor;
}

__device__ Vec3f _phong(const Vec3f& dir
		, ObjectBox& obj
		, DevObject *hitObject
		, const Vec3f& hitPoint
		, const Vec2f& hitTexCoordinates
		, GPU::RenderingContext& rctx){
	Vec3f diffuse, specular;
	Vec3f hitColor;
    float _pattern = 0, shadow;
	for (uint32_t i = 0; i < obj.lightsNumber; ++i){
		bool visible = isPointVisible(obj, i, hitPoint, rctx, rctx.bias);
		// compute the diffuse component
		shadow = (visible) ? 1 : 0.6;
		diffuse += shadow * hitObject->appearence.albedo * rctx.lightIntensity * fmaxf(0.3f, rctx.hitNormal.dot(-rctx.lightDir));
		specular += shadow *rctx.lightIntensity * powf(fmaxf(0.3f, reflect(rctx.lightDir, rctx.hitNormal).dot( -dir ) ), hitObject->appearence.specularExponent);
		}
	float *pattern_ptr = nullptr;
	pattern(hitObject, hitTexCoordinates, _pattern, &pattern_ptr);
	return diffuse * hitObject->appearence.diffuseWeight * colorify(pattern_ptr, hitObject->appearence.colors ) + specular * hitObject->appearence.specularWeight;
}

__device__
Vec3f GPU::shade(
     Ray &ray,
     ObjectBox& obj,
    const Camera &options
    )
{
    RenderingContext rctx;
	Vec3f hitColor = 0;
    DevObject *hitObject = nullptr;

	if( GPU::trace(ray, obj, rctx, &hitObject) ){

			//here we perform shading
			Vec3f hitPoint (ray.orig + ray.dir * ray.tNearest);
			Vec2f hitTexCoordinates;
			hitObject->properties(*hitObject, hitPoint, rctx, hitTexCoordinates);

			float specular = 1;
			if(hitObject->appearence.material == REFLECTION)
			{
                bool outside = ray.dir.dot(rctx.hitNormal) < 0;
                Vec3f bias(rctx.hitNormal * rctx.bias);
                Vec3f reflectionRayOrig(outside ? hitPoint + bias : hitPoint - bias);
				Ray reflectedRay (reflectionRayOrig, reflect(ray.dir, rctx.hitNormal).normalize() );
				specular = hitObject->appearence.specularWeight;
				DevObject *_hitObject = nullptr;
				if ( GPU::trace(reflectedRay, obj, rctx, &_hitObject) ){
					hitPoint = reflectedRay.orig + reflectedRay.dir * reflectedRay.tNearest;
					_hitObject->properties(*_hitObject, hitPoint, rctx, hitTexCoordinates);
					hitObject = _hitObject;
				}
			}
			hitColor = specular*_phong(ray.dir, obj, hitObject, hitPoint, hitTexCoordinates, rctx);

	}else{
    	hitColor = obj.backgroundColor;
    }

	return hitColor;
}
__global__ void raytracer( Camera cam
		, ObjectBox objects
	, unsigned char * dev_bitmap ){
    // map from threadIdx/BlockIdx to pixel position
#ifdef __ASYNCH
    int i = cam.xOffset() + threadIdx.x + blockIdx.x * blockDim.x;
    int j = cam.yOffset() + threadIdx.y + blockIdx.y * blockDim.y;

    int offset = i + j * cam.width();
#else

     int i = threadIdx.x + blockIdx.x * blockDim.x;
     int j = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = i + j * blockDim.x * gridDim.x;
#endif

    if( i < (cam.xOffset() + cam.partialWidth() ) && j < (cam.yOffset() + cam.partialHeight() ) ){

	 float x =(2 * (i + 0.5) / (float)cam.width() - 1) * cam.aspect() * cam.scale();
	 //float y = (1 - 2 * (j + 0.5) / (float)cam.height_) * cam.scale;
	 float y = (1 - 2 * (float(cam.height() - j) - 0.5) //because OPENGL has an origin at the low left corner
	 		/ (float)cam.height()) * cam.scale();

    Vec3f origin;
    Vec3f direction;
    cam.modelOrig.mulPointMat(Vec3f(0), origin);
    cam.modelDir.mulVecMat(Vec3f(x, y, -1), direction);
    direction.normalize();
    Ray ray(origin, direction);
    Vec3f colors(  GPU::shade( ray, objects, cam)  );
#ifdef __TEXTURED_OPENGL
    __shared__ float4 color;
    unsigned int shared_offset = 4 * (blockDim.x * threadIdx.y +  threadIdx.x);

     color.x =  clamp(0.0f, 1.0f, colors.xyz.x);
     color.y =  clamp(0.0f, 1.0f, colors.xyz.y);
     color.z =  clamp(0.0f, 1.0f, colors.xyz.z);
     color.w = 1.0f;

    __syncthreads();
    surf2Dwrite(color, image, 4*offset * sizeof(color), y, cudaBoundaryModeClamp);

#else
    __shared__ char  sharedBuffer[ THREADS_ALONG_X * THREADS_ALONG_X * 4 ];
    unsigned int shared_offset = 4 * (blockDim.x * threadIdx.y +  threadIdx.x);

    sharedBuffer[shared_offset]     = static_cast< char>(255 * clamp(0, 1, colors.xyz.x));
    sharedBuffer[shared_offset + 1] = static_cast< char>(255 * clamp(0, 1, colors.xyz.y));
    sharedBuffer[shared_offset + 2] = static_cast< char>(255 * clamp(0, 1, colors.xyz.z));
    sharedBuffer[shared_offset + 3] = static_cast< char>(255);

    __syncthreads();

    for(int k = 0; k < 4; ++k)
    	dev_bitmap[4*offset + k] = sharedBuffer[shared_offset + k];
    }
#endif
}
