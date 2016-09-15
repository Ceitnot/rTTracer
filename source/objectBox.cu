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
#include "objectBox.cuh"
Mesh::Mesh(
	        const uint32_t nfaces
	        ,const std::vector<uint32_t> &faceIndex
	        ,const std::vector<uint32_t> &vertsIndex
	        ,std::vector<Vec3f> &verts
	        ,const std::vector<Vec3f> &normals
	        ,const std::vector<Vec2f> &st
	        , const SquareMatrix4f & model) :
	        numTris(0)
	   	   , vertsNumber( verts.size() )
	   {
	       uint32_t k = 0;
	       // count the maximum number of polygons
	       for (uint32_t i = 0; i < nfaces; ++i) {
	           numTris += faceIndex[i] - 2;
	         //count the maximum number of vertices
	           k += faceIndex[i];
	       }
	      // allocate memory to store the position of the mesh vertices on device
	       	  allocateGPU(&P, vertsNumber*sizeof(Vec3f));
	       // allocate memory to store triangles indices
	       	  allocateGPU(&triangles, numTris*sizeof(Triangle));
	       	  std::vector<Triangle> _triangles(numTris);
	       	// allocate memory to store normals
	       	  allocateGPU(&N, numTris*3*sizeof(Vec3f));
	       	  std::vector<Vec3f> _N(numTris * 3);
	       	  allocateGPU(&texCoordinates, numTris*3*sizeof(Vec3f));
	       	  std::vector<Vec2f> _texCoordinates(numTris * 3);
	       //triangulation
	       uint32_t l = 0;
	       for (uint32_t i = 0, k = 0; i < nfaces; ++i) { // for each  face
	           for (uint32_t j = 0; j < faceIndex[i] - 2; ++j) { // for each triangle in the face
	        	   //set vertices and their indexes
	        	   _triangles[l/3][0] = vertsIndex[k];
	        	   _triangles[l/3][1] = vertsIndex[k + j + 1];
	        	   _triangles[l/3][2] = vertsIndex[k + j + 2];
	               _N[l] = normals[k];
	               _N[l + 1] = normals[k + j + 1];
	               _N[l + 2] = normals[k + j + 2];
	               _texCoordinates[l] = st[k];
	               _texCoordinates[l + 1] = st[k + j + 1];
	               _texCoordinates[l + 2] = st[k + j + 2];
	               l += 3;
	           }
	           k += faceIndex[i];
	       }
	       //transform object to world space
	       thrust::host_vector<Vec3f> host( vertsNumber );
	       for(uint32_t i = 0; i < vertsNumber; ++i){
	    	   model.mulPointMat(verts[i], host[i]);
	       }
	       //copy transformed points to device
	       thrust::device_vector<Vec3f> device(host);
	       //extract  a raw pointer
	       P = thrust::raw_pointer_cast( device.data() );
	       //copyToDevice(P, verts.data(), greatestIndex*sizeof(Vec3f));
	       copyToDevice(triangles, _triangles.data(), numTris*sizeof(Triangle));
	       copyToDevice(N, _N.data(), numTris*3*sizeof(Vec3f));
	       copyToDevice(texCoordinates, _texCoordinates.data(), numTris*3*sizeof(Vec3f));
	   }
	 Mesh::Mesh():
		   vertsNumber(0)
			, numTris(0)// number of triangles
			, P (NULL)
			,triangles (NULL)
			,N(NULL)// triangles vertex normals
			,texCoordinates(NULL)
							{}
	__host__ __device__
	  const Vec3f& Mesh::operator[](size_t i) const {
		   return P[i];
	   }
	__host__ __device__
	   Vec3f& Mesh::operator[](size_t i)  {
		   return const_cast<Vec3f&>
		   ( static_cast<const Mesh *>(this)->operator [](i) );
	   }
	    Mesh::~Mesh(){
	    	checkAndNull( cudaFree(P) , P );
	    	checkAndNull( cudaFree(triangles) , triangles );
	    	checkAndNull( cudaFree(N) , N );
	    	checkAndNull( cudaFree(texCoordinates) , texCoordinates );
	    }
	    Vec3f* Mesh::data(){
	    	return P;
	    }

	    __host__ __device__
	    Sphere::Sphere(const SquareMatrix4f &model, const float &radius, Vec3f (&planeNormal)[NORMALS] ):
	    						_radius(radius)
	    					  , _squareRadius(radius *radius){
	    	model.mulPointMat(Vec3f(0), _center);
	    	planeTangentPoint( planeNormal );
	    }
	    __host__ __device__
	    Sphere::Sphere(const Sphere& sphere):
					_radius(sphere._radius)
					, _squareRadius(sphere.squareRadius())
					,_center(sphere._center){
	    	for(int i = 0; i < NORMALS * NORMAL_SET; ++i){
				tangentDistances[i] = sphere.tangentDistances[i];
	    	}
	    }
	    __host__ __device__
	    Sphere::Sphere(const Vec3f& center
	    		, const float &radius
	    		, Vec3f (&planeNormal)[NORMALS] ):
									_center(center)
									,_radius(radius)
	    							, _squareRadius(radius *radius)
	    {
	    	planeTangentPoint( planeNormal );
	    }
	    __host__ __device__
	    float Sphere::radius() const{
			return _radius;
	    }
	    __host__ __device__
	    const float& Sphere::operator[](uint8_t i) const {
	    	return tangentDistances[i];
	    }
	    __host__ __device__
	     float& Sphere::operator[](uint8_t i){
	    	return const_cast<float&>
	    	( static_cast<const Sphere *>(this)->operator [](i) );
	    }
	   __host__ __device__
	    float Sphere::squareRadius()const{
			return _squareRadius;
	    }

	  __host__ __device__
	   const Vec3f& Sphere::center() const {
			return _center;
	    }
	  __host__ __device__
	    Vec3f& Sphere::center(){
		  return const_cast<Vec3f&>(
				  static_cast<const Sphere*>(this)->center() );
	    }
	  void Sphere::planeTangentPoint( Vec3f (&planeNormal)[NORMALS] ) {
		  for(int i = 0 ; i < NORMALS; ++i){
			  tangentDistances[2*i] =(_center - ( planeNormal[i].normalize()*_radius  ) ).dot(planeNormal[i]);
			  tangentDistances[2*i + 1] = (_center + ( planeNormal[i]*_radius  ) ).dot(planeNormal[i]);
		  }
	}
 	  size_t Sphere::bytes()const{
 		  return sizeof(*this);
 	  }
 	  size_t Sphere::getVertsNumber()const{
 		  return vertsNumber;
 	  }
	  __host__ __device__
	  Sphere::~Sphere(){}

	void* Sphere::getObject(){
				return reinterpret_cast<void *>(this);
			}

	__host__ __device__
	Plane::Plane(): dimentions( make_float2( 0, 0) ), faceNormal(){}
 	__host__ __device__
 	Plane::Plane(const Plane& plane):
 		  dimentions( make_float2(plane.dimentions.x, plane.dimentions.y) )
 		, faceNormal(plane.faceNormal){
 		for(int i = 0; i < vertsNumber; ++i){
 			points[i] = plane.points[i];
 		}
 	}
 	__host__ __device__
 	Plane::Plane(const Vec3f& first
 		 ,const Vec3f& second
 		 ,const Vec3f& third
 		 ,const Vec3f& normal){
 		points[0] = first;
 		points[1] = second;
 		points[2] = third;
 		points[3] = Vec3f(points[2].xyz.x - points[1].xyz.x + points[0].xyz.x
 						, points[2].xyz.y - points[1].xyz.y + points[0].xyz.y
 						, points[2].xyz.z - points[1].xyz.z + points[0].xyz.z);

 		//find face normal
 		faceNormal = normal;
 		dimentions.x = (points[0] - points[1]).length();
 		dimentions.y = (points[0] - points[3]).length();
 	}
 	__host__ __device__
 	Plane::Plane(const Vec3f& first
 	 		 ,const Vec3f& second
 	 		 ,const Vec3f& third){

 	 		points[0] = first;
 	 		points[1] = second;
 	 		points[2] = third;
 	 		points[3] = Vec3f(points[2].xyz.x - points[1].xyz.x + points[0].xyz.x
 	 						, points[2].xyz.y - points[1].xyz.y + points[0].xyz.y
 	 						, points[2].xyz.z - points[1].xyz.z + points[0].xyz.z);

 	 		//find face normal
 	 		faceNormal = (points[0] - points[1]).cross(points[3] - points[1]);
 	 		faceNormal.normalize();
 	 		dimentions.x = (points[0] - points[1]).length();
 	 		dimentions.y = (points[0] - points[3]).length();
 	 	}
 	__host__ __device__
 	const Vec3f& Plane::operator[](uint8_t i) const {
 		return points[i];
 	}
 	__host__ __device__
 	Vec3f& Plane::operator[](uint8_t i){
 		return const_cast<Vec3f&>(
 				  static_cast<const Plane*>(this)->operator [](i) );
 	}
 	__host__ __device__ Plane& Plane::operator= (const Plane& plane){
 		Plane(plane).swap(*this);
 		return *this;
 	}
 	__host__ __device__
 	const Vec3f& Plane::normal()const{
 		return faceNormal;
 	}
 	__host__ __device__
 	const float& Plane::width()const{
 		return dimentions.x;
 	}
 	__host__ __device__
 	const float& Plane::height()const{
 		return dimentions.y;
 	}
 	__host__ __device__ void Plane::swap(Plane& plane){
 		algorithms::swap(dimentions.x, plane.dimentions.x);
 		algorithms::swap(dimentions.y, plane.dimentions.y);
 		faceNormal.swap(plane.faceNormal);
 		for(int i = 0; i < vertsNumber; ++i){
 			points[i].swap(plane.points[i]);
 		}
 	}
 	__host__ __device__
	 Plane::~Plane(){}

	void* Plane::getObject(){
				return reinterpret_cast<void *>(this);
			}
	  size_t Plane::bytes()const{
		  return sizeof(*this);}
 	  size_t Plane::getVertsNumber()const{
 		  return vertsNumber;
 	  }
 	__host__ __device__
 	Cube::Cube(const Vec3f &vmin, const Vec3f &vmax)
     {
 		//the closer down left
 		verts[0] = vmin;
 		//the closer down right
 		verts[1].xyz.x = vmax.xyz.x;
 		verts[1].xyz.y = vmin.xyz.y;
 		verts[1].xyz.z = vmin.xyz.z;
 		//the closer upper right
 		verts[2].xyz.x = vmax.xyz.x;
 		verts[2].xyz.y = vmax.xyz.y;
 		verts[2].xyz.z = vmin.xyz.z;
 		//the closer upper left
 		verts[3].xyz.x = vmin.xyz.x;
 		verts[3].xyz.y = vmax.xyz.y;
 		verts[3].xyz.z = vmin.xyz.z;
 		//the far down right
 		verts[4].xyz.x = vmax.xyz.x;
 		verts[4].xyz.y = vmin.xyz.y;
 		verts[4].xyz.z = vmax.xyz.z;
 		//the far down left
 		verts[5].xyz.x = vmin.xyz.x;
 		verts[5].xyz.y = vmin.xyz.y;
 		verts[5].xyz.z = vmax.xyz.z;
 		//the far upper left
 		verts[6].xyz.x = vmin.xyz.x;
 		verts[6].xyz.y = vmax.xyz.y;
 		verts[6].xyz.z = vmax.xyz.z;
 		//the far upper right
 		verts[7] = vmax;

 		vertsIndex[0][0] = 0;
 		vertsIndex[0][1] = 1;
 		vertsIndex[0][2] = 2;
 		vertsIndex[0][3] = 3;

 		vertsIndex[1][0] = 2;
 		vertsIndex[1][1] = 7;
 		vertsIndex[1][2] = 6;
 		vertsIndex[1][3] = 3;

 		vertsIndex[2][0] = 7;
 		vertsIndex[2][1] = 4;
 		vertsIndex[2][2] = 5;
 		vertsIndex[2][3] = 6;

 		vertsIndex[3][0] = 4;
 		vertsIndex[3][1] = 1;
 		vertsIndex[3][2] = 0;
 		vertsIndex[3][3] = 5;

 		vertsIndex[4][0] = 0;
 		vertsIndex[4][1] = 3;
 		vertsIndex[4][2] = 6;
 		vertsIndex[4][3] = 5;

 		vertsIndex[5][0] = 4;
 		vertsIndex[5][1] = 7;
 		vertsIndex[5][2] = 2;
 		vertsIndex[5][3] = 1;

 		//normals
 		for(uint8_t i = 0; i < planes; ++i){
 			normals[i] = computeNormal(i);
 		}
     }
     __host__ __device__
      const Vec3f& Cube::operator[](uint8_t i)const{
     	 return verts[i];
     	}
     __host__ __device__
       Vec3f& Cube::operator[](uint8_t i){
     	return const_cast<Vec3f&>(
 			static_cast<const Cube*>(this)->operator [](i));
     	}

     __host__ __device__
     	const Vec3f& Cube::min()const{
     		return verts[0];
     }
     __host__ __device__
     const Vec3f& Cube::max()const{
     	return verts[7];
     }
     __host__ __device__
     	const Vec3f& Cube::normal(uint8_t planeIndex) const {
     		return normals[planeIndex];
     }
     __host__ __device__
      int Cube::vIndex(int planeIndex, int element) const {
     	assert(element < 4);
     	assert(planeIndex >= 0);
     	return vertsIndex[planeIndex][element];
     }
   __host__ __device__ Cube::~Cube(){}
	void* Cube::getObject(){
				return reinterpret_cast<void *>(this);
			}
	  size_t Cube::bytes()const{
		  return sizeof(*this);
	  }
 	  size_t Cube::getVertsNumber()const{
 		  return vertsNumber;
 	  }
__host__ __device__
DevObject::DevObject( const DevObject & object)
	: _data(object._data)
	, _boundaries(object._boundaries)
	, vertsNumber(object.vertsNumber)
	, appearence(appearence)
	, intersect(object.intersect)
	, intersectBV(object.intersectBV)
	, properties(object.properties)
	, transform(object.transform)
	, setBV(object.setBV)
	, pattern(object.pattern)
	, destructor(object.destructor)
	{}
	__host__ __device__
	 DevObject::DevObject(uint32_t vertsNumber, const Appearence& appearence)
		:_data(nullptr)
		, _boundaries(nullptr)
		, vertsNumber(vertsNumber)
		, appearence(appearence)
		, intersect(nullptr)
		, intersectBV(nullptr)
		, properties(nullptr)
		, transform(nullptr)
		, setBV(nullptr)
		, pattern(nullptr)
		, destructor(nullptr)
		{}
	__host__ __device__
	DevObject::DevObject(const Appearence& appearence):
						_data(nullptr)
					  , _boundaries(nullptr)
					  , vertsNumber(0)
					  , appearence(appearence)
					  , intersect(nullptr)
	                  , intersectBV(nullptr)
	                  , properties(nullptr)
	                  , transform(nullptr)
					  , setBV(nullptr)
					  , pattern(nullptr)
	                  , destructor(nullptr)
	                  {}
	__host__ __device__
	DevObject& DevObject::operator=(const DevObject& object){
		DevObject tmp(object);
		algorithms::swap(*this, tmp);
		return *this;
	}
	 __host__ __device__
	const void* DevObject::data() const {
		return _data;
	}
	 __host__ __device__
	 void* DevObject::data() {
		 return const_cast<void *>(
				 static_cast<const DevObject*>(this)->data() );
	}
	 __host__ __device__
	const void* DevObject::boundaries() const{
		 return _boundaries;
	 }
	 __host__ __device__
	 void DevObject::swap(DevObject& object){
		 algorithms::swap<void*>(object._boundaries, _boundaries);
		 algorithms::swap<void*>(object._data, _data);
		 algorithms::swap<Appearence>(object.appearence, appearence);
		 algorithms::swap<Intersect>(object.intersect, intersect);
		 algorithms::swap<IntersectBV>(object.intersectBV, intersectBV);
		 algorithms::swap<SquareMatrix4f>(object.objectToWorld, objectToWorld);
		 algorithms::swap<Properties>(object.properties, properties);
		 algorithms::swap<SetBV>(object.setBV, setBV);
		 algorithms::swap<Transform>(object.transform, transform);
		 algorithms::swap<uint32_t>(object.vertsNumber, vertsNumber);
	 }

	 __host__ __device__
	 DevObject::~DevObject(){ }

	 __host__ __device__
	 const Vec3f& DevObject::albedo() const{
		 return appearence.albedo;
	 }


	 ObjectBox::ObjectBox( std::vector<DevObject>& dVec) {
		loadNormals();
		allocateGPU( &_array, dVec.size()*sizeof(DevObject) );

		DevObject* tmp_obj = dVec.data();
		copyToDevice(_array, tmp_obj, dVec.size()*sizeof(DevObject));
		_size  = dVec.size();
	}
	__host__ __device__
	ObjectBox::ObjectBox(const ObjectBox & karr)
	:_array(karr._array)
	,_size(karr._size)
	, lights(karr.lights)
	,lightsNumber(karr.lightsNumber)
	, backgroundColor(karr.backgroundColor)
	{
		loadNormals();
	}
	__host__ __device__
	ObjectBox::ObjectBox()
	:_array(NULL)
	,_size(0)
	,backgroundColor(0.3){
		loadNormals();
	}
	__host__ __device__
	ObjectBox::ObjectBox(DevObject* objects, int size)
	:_array(objects)
	,_size(size){
		loadNormals();
	}
	//deletes inside Object array
	void ObjectBox::destroy(){
		cudaError_t err = cudaFree(_array);
		if(err != cudaSuccess){
			std::cout<<cudaGetErrorString(err)<<" in "<<__FILE__<<" at line "<<__LINE__;
			std::cout<<std::endl;
		}else{
			_array = nullptr;
		}
		 err = cudaFree(lights);
		//cudaError_t err;
		if(err != cudaSuccess){
			std::cout<<cudaGetErrorString(err)<<" in "<<__FILE__<<" at line "<<__LINE__;
			std::cout<<std::endl;
		}else{
			lights = nullptr;
		}
	}
	__host__ __device__
	DevObject* ObjectBox::data(){
		return _array;
	}
	ObjectBox::~ObjectBox(){}
	__host__ __device__ DevObject& ObjectBox::operator [] (uint8_t i) { return _array[i]; }
	__device__
	 bool ObjectBox::hit(Ray &ray
			 , DevPriorityQueue<DevObject *, MAX_OBJECTS>& queue
			 , GPU::RenderingContext& rctx){
	    bool hit = false;

	    //[0] - numerator [1] - denominator
	    float precompute[2][ Boundaries::nNormals ];
	    for (uint8_t i = 0; i < Boundaries::nNormals; ++i) {
	    	precompute[0][i] = boundingPlaneNormals[i].dot(ray.orig);
	    	precompute[1][i] = boundingPlaneNormals[i].dot(ray.dir);
	    }
		for (uint32_t k = 0; k < _size; ++k) {
			uint8_t normalIndex = 0;
			if( _array[k].intersectBV( ray, precompute, _array[k], normalIndex) )
			{
				queue.push(ray.t, &_array[k]);
				//queue.push(ray.tNearest, &_array[k]);
				hit = true;
			}
			ray.tNearest = kInfinity;
			ray.t = -kInfinity;
		}
		return hit;
	}
	__host__ __device__
	 ILight& ObjectBox::getlight(uint8_t lightNumber){
		return lights[lightNumber];
	}

	__host__ __device__
	void ObjectBox::loadNormals(){
		boundingPlaneNormals[0] = Vec3f(1, 0, 0);
		boundingPlaneNormals[1] = Vec3f(0, 1, 0);
		boundingPlaneNormals[2] = Vec3f(0, 0, 1);
		boundingPlaneNormals[3] = Vec3f( sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f);
		boundingPlaneNormals[4] = Vec3f(-sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f);
		boundingPlaneNormals[5] = Vec3f(-sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f);
		boundingPlaneNormals[6] = Vec3f( sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f);
	}
	__host__ __device__
	Appearence::Appearence():
							material(DIFFUSE)
						  , refractionIndex(0)
						  , albedo(0)
	  	  	  	  	  	  , texture(nullptr)
						  , isSmooth(false)
						  , diffuseWeight(0)
						  , specularWeight(0)
						  , specularExponent(0){
		colors[0] = Vec3f(0);
		colors[1] = Vec3f(0);
	}
	 __host__ __device__
	 Appearence::Appearence(const Vec3f& color
			 	 	 	 , const float& diffuseWeight
						 , const float& specularWeight
						 , const float& specularExponent
						 , const char* texture):
						   material(PHONG)
						 , refractionIndex(0)
						 , albedo(0.2f)
	 	 	 	 	 	 , texture(texture)
						 , isSmooth(true)
						 , diffuseWeight(diffuseWeight)
	 	 	 	 	 	 , specularWeight(specularWeight)
	 	 	 	 	 	 , specularExponent(specularExponent){
			colors[0] = color;
			colors[1] = Vec3f(0);
	 }

	__host__ __device__
	Appearence::Appearence(const Appearence& appearence)
							: material(appearence.material)
							, refractionIndex(appearence.refractionIndex)
							, texture(appearence.texture)
							, isSmooth(appearence.isSmooth)
							, albedo(appearence.albedo)
							, diffuseWeight(appearence.diffuseWeight)
							, specularWeight(appearence.specularWeight)
							, specularExponent(appearence.specularExponent){
		colors[0] = appearence.colors[0];
		colors[1] = appearence.colors[1];
	}

	__host__ __device__
	Appearence::Appearence(const Vec3f& color
				  , const MateryalType& material
				  , const float& refractionIndex
				  , const float& albedo
				  , const char* texture
				  , const bool& isSmooth)
		 	 	 	 : material(material)
		 	 	 	 , refractionIndex(refractionIndex)
		 	 	 	 , albedo(albedo)
					 , texture(texture)
		 	 	 	 , isSmooth(isSmooth)
					 , diffuseWeight(0.8f)
					 , specularWeight(0.2f)
					 , specularExponent(10.0f){
		colors[0] = color;
		colors[1] = Vec3f(0);
	}
	__host__ __device__
	Appearence::Appearence(const float& specularWeight)
		 	 	 	 : material(REFLECTION)
		 	 	 	 , refractionIndex(0)
		 	 	 	 , albedo(0.5f)
					 , texture("none")
		 	 	 	 , isSmooth(true)
					 , diffuseWeight(0.8f)
					 , specularWeight(specularWeight)
					 , specularExponent(3.0f){
		colors[0] = Vec3f(1);
		colors[1] = Vec3f(1);
	}
	__host__ __device__
	Appearence::Appearence(const Vec3f& color
						 , const MateryalType& material)
		 	 	 	 : material(material)
		 	 	 	 , refractionIndex(0)
		 	 	 	 , albedo(0.18)
					 , texture("grid")
		 	 	 	 , isSmooth(true)
					 , diffuseWeight(0.8f)
	 	 	 	 	 , specularWeight(0.2f)
	 	 	 	 	 , specularExponent(10.0f){
		colors[0] = color;
		colors[1] = Vec3f(0);
	}
	__host__ __device__
	Appearence::Appearence(const Vec3f& color
				  , const MateryalType& material
				  , const float& albedo
				  , const char* texture
				  , const bool& isSmooth)
		 	 	 	 : material(material)
		 	 	 	 , refractionIndex(0)
		 	 	 	 , albedo(albedo)
					 , texture(texture)
		 	 	 	 , isSmooth(isSmooth)
					 , diffuseWeight(0.8f)
					 , specularWeight(0.2f)
					 , specularExponent(10.0f){

		colors[0] = color;
		colors[1] = Vec3f(0);
	}
	__host__ __device__
	Appearence& Appearence::operator=(const Appearence& appearence){
		Appearence(appearence).swap( *this );
		return *this;
	}
	__host__ __device__
	void Appearence::swap(Appearence& appearence){
		algorithms::swap(material, (appearence.material));
		algorithms::swap(refractionIndex, appearence.refractionIndex);
		algorithms::swap(albedo, appearence.albedo);
		algorithms::swap(isSmooth, appearence.isSmooth);
		colors[0].swap(appearence.colors[0]);
		colors[1].swap(appearence.colors[1]);
		algorithms::swap(diffuseWeight, appearence.diffuseWeight);
		algorithms::swap(specularExponent, appearence.specularExponent);
		algorithms::swap(specularWeight, appearence.specularWeight);
		algorithms::swap(texture, appearence.texture);
	}
	 __host__ __device__
	 Appearence& Appearence::setTextureColor(const Vec3f& color){
		 colors[1] = color;
		 return *this;
	 }
	 __host__ __device__
	 Appearence& Appearence::setObjectColor(const Vec3f& color){
		 colors[0] = color;
		 return *this;
	 }
	 __host__ __device__
	 float2 texCoordinates(const Plane& plane, const Vec3f& hitPoint){
	 	float2 uv;
	 	uv.x = height(hitPoint, plane[0], plane[1]);
	 	uv.y = height(hitPoint, plane[3], plane[0]);
	 	return uv;
	 }
