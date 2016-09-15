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
#ifndef OBJECTBOX_CUH_
#define OBJECTBOX_CUH_
#include "algorithms.cuh"
#include "light.cuh"
#include "acceleration.cuh"
#include "thrustHelper.cuh"
#include <unordered_map>
class DevObject;
class ObjectBox;
/*!
\addtogroup device_pointers Device static function pointers
\detail - Function pointer typedefs.
@{
 */
typedef bool (*Intersect)(Ray&, DevObject&, GPU::RenderingContext& );
typedef bool (*IntersectBV)(Ray& ray, float (&precompute)[2][7], DevObject& currentObject, uint8_t& normalIndex);

typedef void (*Properties)(const DevObject& object,
        const Vec3f &hitPoint,
       // const Vec3f &viewDirection,
        GPU::RenderingContext& ctx,
        Vec2f &hitTextureCoordinates);

 typedef void (*Transform)(DevObject* , const SquareMatrix4f& );
 typedef void (*SetBV) (const DevObject*, Boundaries*, Vec3f (&)[7], float * (&)[7] );
 typedef float (*Pattern)(const Vec2f&);
 /*!
 @}
  */

 typedef std::unordered_map <std::string,  Pattern *> textureMap;

/*!
  \brief The class stores device pointers to function
  That's a part of polymorphic architecture.
 \addtogroup wrapping_and_description Wrapping and description of the objects
    @{
*/

 struct Polymorphic{
	 /*!
	  *\param[in] dev_intersect - pointer to the intersection test function
	  *\param[in] dev_properties - pointer to the function which gets normal, texture coordinates, etc.
	  *\param[in] dev_intersectBV - bounding volume intersect test function pointer
	  *\param[in] dev_transform - pointer to the object transformation function
	  *\param[in] setBV - pointer to the bounding volume set up function
	  *\param[in] dev_pattern - pointer to the texturing function
	  * */
	 Polymorphic(
			 	  const Intersect& dev_intersect  = nullptr
			 	, const Properties& dev_properties  = nullptr
			 	, const IntersectBV& dev_intersectBV  = nullptr
			 	, const Transform& dev_transform = nullptr
			 	, const SetBV& setBV  = nullptr
			 	, const Pattern& dev_pattern = nullptr
	 ):intersect(dev_intersect)
	 , properties(dev_properties)
	 , intersectBV(dev_intersectBV)
	 , transform(dev_transform)
	 , setBV(setBV)
	 , pattern(dev_pattern){}
	 const Intersect& intersect;
	 const Properties& properties;
	 const IntersectBV& intersectBV;
	 const Transform& transform;
	 const SetBV& setBV;
	 const Pattern& pattern;
 };
///Each of the implemented shading type.
 enum MateryalType{
	 	 	 	 	 DIFFUSE, ///< a minimum glossiness effect
	 	 	 	 	 REFLECTION,///< a mirror surface
	 	 	 	 	 PHONG ///< glossiness and shininess being adjusted
 	 	 	 	 };
 /// Textural setup.
 enum ProcediralTexture {
	 	 	 	 	 	 NONE, ///< no texture
	 	 	 	 	 	  WAVE, ///< blurred wave pattern
	 	 	 	 	 	  STRIPS, ///< lines/strips pattern
	 	 	 	 	 	  GRID, ///< blurred grid pattern
	 	 	 	 	 	  CHECKER ///< checker pattern
 	 	 	 	 	 	 };
/*!
 * \brief Class for textural, color and shading detail setups.*/
 struct Appearence{

	 MateryalType material;
	 float refractionIndex;
	 Vec3f albedo;

	 //texturing
	 const char* texture;

	 bool isSmooth;
	 Vec3f colors[2]; ///< 0 is object color, 1 is texture color
	 // phong model
	 float diffuseWeight;
	 float specularWeight; ///  specular weight
	 float specularExponent;   ///  specular exponent

	 __host__ __device__
	 Appearence();
	 __host__ __device__
	 Appearence(
			   const Vec3f& color
			 , const float& diffuseWeight
			 , const float&  specularWeight
			 , const float&  specularExponent
			 , const char* texture = nullptr);
	 __host__ __device__
	 Appearence(const Appearence& appearence);
	 __host__ __device__
	 Appearence( const Vec3f& color
			  , const MateryalType& material
			  , const float& refractionIndex
			  , const float& albedo = 0.18f
			  , const char* texture = nullptr
			  , const bool& isSmooth = true);
	 __host__ __device__
	 Appearence(   const Vec3f& color
			 	 , const MateryalType& material);
	__host__ __device__
	Appearence(const float& specularWeight);
	 __host__ __device__
	 Appearence(const Vec3f& color
			  , const MateryalType& material
			  , const float& albedo
			  , const char* texture = nullptr
			  , const bool& isSmooth = true);
	 __host__ __device__
	 Appearence& operator=(const Appearence& appearence);
	 __host__ __device__
	 void swap(Appearence& appearence);
	 __host__ __device__
	 Appearence& setTextureColor(const Vec3f& color);
	 __host__ __device__
	 Appearence& setObjectColor(const Vec3f& color);
 };

/*!
 * \brief Parent class for most of the class objects in the project except of Triangle class
 * */
 class HostObject{
 public:
 	virtual Polymorphic getDevicePolymorphic(const Pattern& dev_pattern) const = 0;
 	virtual void * getObject() = 0;
 	virtual size_t bytes() const = 0;
 	virtual size_t getVertsNumber()const = 0;
 	__host__ __device__
 	virtual ~HostObject(){};
 };
/*!
 * \brief Class for triangulated mesh polygons.
 * */
 class Mesh
 {
 public:
     // Build a triangle mesh from a face index array and a vertex index array

    explicit Mesh(
         const uint32_t nfaces
         ,const std::vector<uint32_t> &faceIndex
         ,const std::vector<uint32_t> &vertsIndex
         ,std::vector<Vec3f> &verts
         ,const std::vector<Vec3f> &normals
         ,const std::vector<Vec2f> &st
         ,const SquareMatrix4f & model);

 	   explicit Mesh();
 	__host__ __device__
 	  const Vec3f& operator[](size_t i) const;
 	__host__ __device__
 	  Vec3f& operator[](size_t i);
 	  Polymorphic getDevicePolymorphic(const Pattern& dev_pattern);
 	  Vec3f* data();
 	 ~Mesh();
     uint32_t vertsNumber; ///< the maximum number of vertices
     uint32_t numTris;        ///< number of triangles
     Vec3f * P ;
     Triangle * triangles ;
     Vec3f * N;              ///< triangles vertex normals
     Vec2f * texCoordinates; ///< triangles texture coordinates
   };

 class Sphere: public HostObject{

 public:
 	    static const size_t vertsNumber = NORMALS * NORMAL_SET;

 	    __host__ __device__
 	    Sphere(const SquareMatrix4f &model, const float &radius, Vec3f (&planeNormal)[NORMALS] );
 	    __host__ __device__
 	    Sphere(const Vec3f& center
 	    		, const float &radius
 	    		, Vec3f (&planeNormal)[NORMALS] );
 	    __host__ __device__
 	    Sphere(const Sphere& sphere);
 	    __host__ __device__
 	    const float& operator[](uint8_t i) const;
 	    __host__ __device__
 	    float& operator[](uint8_t i);
 		 __host__ __device__
 		 float radius() const;
 		 __host__ __device__
 		 float squareRadius()const;
 		 __host__ __device__
 		  const Vec3f& center() const;
 		 __host__ __device__
 		    Vec3f& center();
 		 __host__ __device__
 		 void printPoints(int number)const;
 	    __host__ __device__
 	    void planeTangentPoint( Vec3f (&planeNormal)[NORMALS] );
 	 	 Polymorphic getDevicePolymorphic(const Pattern& dev_pattern) const;
 	 	void* getObject();
 	 	size_t bytes() const;
 	 	size_t getVertsNumber()const;
 	 	__host__ __device__ ~Sphere();
 private:
 		float _radius, _squareRadius;
 	    Vec3f _center;
 	    float tangentDistances[ vertsNumber ];
 };

 class Plane: public HostObject{
 public:
 	static const size_t vertsNumber = 4;
 	__host__ __device__ Plane();
 	__host__ __device__
 	Plane(const Plane& plane);
 	__host__ __device__
 	Plane(const Vec3f& first
 		 ,const Vec3f& second
 		 ,const Vec3f& third
 		 ,const Vec3f& normal);
 	__host__ __device__
 	Plane(const Vec3f& first
 		 ,const Vec3f& second
 		 ,const Vec3f& third);
 	__host__ __device__
 	const Vec3f& operator[](uint8_t i) const;
 	__host__ __device__
 	Vec3f& operator[](uint8_t i);
 	__host__ __device__ Plane& operator= (const Plane& plane);
 	__host__ __device__
 	const Vec3f& normal()const;
 	__host__ __device__
 	const float& width()const;
 	__host__ __device__
 	const float& height()const;
 	__host__ __device__ void swap(Plane& plane);
 	 Polymorphic getDevicePolymorphic(const Pattern& dev_pattern)const;
 	 void* getObject();
 	 size_t bytes()const;
	 size_t getVertsNumber()const;
 	__host__ __device__ ~Plane();
 private:
 	float2 dimentions;
 	Vec3f faceNormal;
 	Vec3f points[vertsNumber];
 };



 class Cube: public HostObject
 {
 public:
 	static const uint8_t vertsNumber = 8;
 	static const uint8_t planes = 6;
 	__host__ __device__
 	Cube(const Vec3f &vmin, const Vec3f &vmax);
     __host__ __device__
      const Vec3f& operator[](uint8_t i)const;
     __host__ __device__
       Vec3f& operator[](uint8_t i);
     __host__ __device__
     	const Vec3f& min()const;
     __host__ __device__
     const Vec3f& max()const;
     __host__ __device__
     	const Vec3f& normal(uint8_t planeIndex) const;
     __host__ __device__
      int vIndex(int planeIndex, int element) const;
 	 Polymorphic getDevicePolymorphic(const Pattern& dev_pattern)const;
 	 void* getObject();
 	 size_t bytes()const;
	 size_t getVertsNumber()const;
 	__host__ __device__ ~Cube();
 private:
     __host__ __device__
     	Vec3f computeNormal(uint8_t planeIndex){
     		Vec3f a( verts[ vertsIndex[planeIndex][1] ] - verts[ vertsIndex[planeIndex][0] ] );
     		Vec3f b( verts[ vertsIndex[planeIndex][2] ] - verts[ vertsIndex[planeIndex][0] ] );
     		return a.cross(b).normalize();
     }
 	Vec3f verts[vertsNumber];
 	int vertsIndex[planes][4];
 	Vec3f normals[planes];
 };

/*!
 * \brief Wrapper class for dynamic polymorphism on the device side.
 * */
class DevObject{
protected:
	void * _data;
	void * _boundaries;
	Color color;
public:
	Appearence appearence;
	uint32_t vertsNumber;
	Intersect intersect;
	IntersectBV intersectBV;
	Properties properties;
	Transform transform;
	SetBV setBV;
	Pattern pattern;
	__host__ __device__
	DevObject( const DevObject & object);
	__host__ __device__
	explicit DevObject(
			 uint32_t vertsNumber
			 , const Appearence& appearence
			 );
	__host__ __device__
	DevObject( const Appearence& appearence );
	__host__ __device__
	DevObject& operator=(const DevObject& object);

template <class ObjectType, class BoundingVolumeType>
	void init(
			  Polymorphic& functions
			 , void (*destr)(DevObject*)
			 , const ObjectType* obj
			 , BoundingVolumeType ** bv
			){
	try{
	    allocateGPU( &_data , sizeof(ObjectType));
	    //allocate space for boundary volumes on GPU for IObject
	    allocateGPU( &_boundaries , sizeof(BoundingVolumeType));
	    //copy pointer to boundaries to Loader's member
	    *bv =
	  const_cast<BoundingVolumeType *>(
	    		reinterpret_cast<const BoundingVolumeType * >( _boundaries )) ;
	    assert(*bv != nullptr);
	    //initialize links to functions

		checkError( cudaMemcpyFromSymbol( &intersect, functions.intersect, sizeof(Intersect) ) );
		checkError( cudaMemcpyFromSymbol( &properties, functions.properties, sizeof(Properties) ) );
		checkError( cudaMemcpyFromSymbol( &intersectBV, functions.intersectBV, sizeof(IntersectBV) ) );
		checkError( cudaMemcpyFromSymbol( &transform, functions.transform, sizeof(Transform) ) );
		checkError( cudaMemcpyFromSymbol( &pattern, functions.pattern, sizeof(Pattern) ) );

		copyToDevice(
	    		reinterpret_cast<ObjectType *>(
	    										const_cast <void*>(_data) )
	    										, obj
	    										, sizeof(ObjectType)
	    										);
	}catch( const std::exception & e){
						throw CudaException(" IObject::init ") + e;

	}
		destructor = destr;
		setBV = functions.setBV;
}

	void init(
			  const HostObject* obj
			 , Boundaries ** bv
			 , Polymorphic& functions
			 , void (*destr)(DevObject*)

			){
	try{
	    allocateGPU( &_data , obj->bytes() );
	    //allocate space for boundary volumes on GPU for IObject
	    allocateGPU( &_boundaries , sizeof(Boundaries));
	    //copy pointer to boundaries to Loader's member
	    *bv =
	  const_cast<Boundaries *>( reinterpret_cast<const Boundaries * >( _boundaries ) ) ;
	    assert(*bv != nullptr);
	    //initialize links to functions

		checkError( cudaMemcpyFromSymbol( &intersect, functions.intersect, sizeof(Intersect) ) );
		checkError( cudaMemcpyFromSymbol( &properties, functions.properties, sizeof(Properties) ) );
		checkError( cudaMemcpyFromSymbol( &intersectBV, functions.intersectBV, sizeof(IntersectBV) ) );
		checkError( cudaMemcpyFromSymbol( &transform, functions.transform, sizeof(Transform) ) );
		checkError( cudaMemcpyFromSymbol( &pattern, functions.pattern, sizeof(Pattern) ) );

		copyToDevice(const_cast <void*>(_data), const_cast<HostObject*>(obj)->getObject(), obj->bytes() );
	}catch( const std::exception & e){
						throw CudaException(" IObject::init ") + e;

	}
		destructor = destr;
		setBV = functions.setBV;
}

	 __host__ __device__
	const void* data() const;
	 __host__ __device__
	void* data();
	 __host__ __device__
	 const Vec3f& albedo()const;
	 __host__ __device__
	const void* boundaries() const;
	 __host__ __device__
	 void swap(DevObject& object);
	void (*destructor)(DevObject*);

	 __host__ __device__
	~DevObject();
	 SquareMatrix4f objectToWorld;
};

class ObjectBox
{
public:
	DevObject* _array;
	ILight* lights;
	int _size, lightsNumber;
	Vec3f boundingPlaneNormals[Boundaries::nNormals];
    Vec3f backgroundColor;
	__host__ __device__
	ObjectBox();
	ObjectBox( std::vector<DevObject>& dVec);
	__host__ __device__
	ObjectBox(const ObjectBox & karr);
	__host__ __device__
	ObjectBox(DevObject* objects, int size);
	__host__ __device__
	DevObject& operator [] (uint8_t i);
	__device__
	 bool hit(Ray &ray
			 , DevPriorityQueue<DevObject *, MAX_OBJECTS>& queue
			 , GPU::RenderingContext& rctx);
	__host__ __device__
	 ILight& getlight(uint8_t lightNumber);
	//deletes inside Object array
	void destroy();
	__host__ __device__
	DevObject* data();
	~ObjectBox();

private:
	__host__ __device__
	void loadNormals();

};

/*! @} */

__host__ __device__
float2 texCoordinates(const Plane& plane, const Vec3f& hitPoint);


#endif /* OBJECTBOX_CUH_ */
