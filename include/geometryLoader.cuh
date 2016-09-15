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
#ifndef GEOMETRYLOADER_CUH_
#define GEOMETRYLOADER_CUH_
#include "objectBox.cuh"
#include "describers.cuh"
#include "parser.cuh"
#include "light.cuh"
/*!
\addtogroup device_pointers Device static function pointers
\ingroup geometry_loading
\detail - The part of the polymorphic behavior implementation on both host and device sides.
@{
 */

/// mesh function
__device__  Intersect   dev_intersect = intersectMesh;
/// mesh function
__device__  IntersectBV dev_intersectBV = intersectBoundingVolume;
/// mesh function
__device__  Properties  dev_properties = meshProperties;
/// mesh function
__device__  Transform  dev_transform = transformMesh;
/// mesh function
  SetBV     set_bv = setMeshAccelerationVolume;

/// plane function
  __device__  Intersect   dev_plane_intersect = intersectPlane;
/// plane function
  __device__  IntersectBV dev_plane_intersectBV = intersectBoundingVolume;
/// plane function
  __device__  Properties  dev_plane_properties = planeProperties;
/// plane function
  __device__  Transform  dev_plane_transform = transformPlane;
/// plane function
    SetBV     set_plane_bv = setPlaneAccelerationVolume;

///cube function
__device__  Intersect   dev_cube_intersect = intersectCube;
///cube function
__device__  IntersectBV dev_cube_intersectBV = intersectBoundingVolume;
///cube function
__device__  Properties  dev_cube_properties = cubeProperties;
///cube function
__device__  Transform  dev_cube_transform = transformCube;
///cube function
  SetBV     set_cube_bv = setCubeAccelerationVolume;

///sphere function
__device__ Intersect dev_sphere_intersect = intersectSphere;
///sphere function
__device__ IntersectBV dev_sphere_intersectBV = intersectBoundingVolume;
///sphere function
__device__ Properties dev_sphere_properties = sphereProperties;
///sphere function
__device__ Transform  dev_sphere_transform = transformSphere;
///sphere function
  SetBV    set_sphere_bv = setSphereAccelerationVolume;

///distant lighting function
__device__  Illuminate   dev_distant_illuminate = illuminateDistant;
///point lighting function
__device__  Illuminate   dev_point_illuminate = illuminatePoint;
///distant lighting function
__device__  ReflectedColor dev_distant_reflected_color = distantLightReflectedColor;
///point lighting function
__device__  ReflectedColor dev_point_reflected_color = distantLightReflectedColor;//<needs to be changed

///texturing pattern function
__device__ Pattern dev_strips = strips;
__device__ Pattern dev_wave = wave;
__device__ Pattern dev_grid = grid;
__device__ Pattern dev_checker = checker;
__device__ Pattern dev_interpolation;
__device__ Pattern dev_none = none;

/*!
	@}
 */
#ifdef __ACCELERATION2
__device__ compare comparator_min = comp_min;
__device__ compare comparator_max = comp_max;
#endif

/*\fn template <class MatrixClass> __global__ void transformation(DevObject* object, MatrixClass mTransform)
 *\brief Device kernel for object transformation.
 *\param[out] - object
 *\param[in] - mTransform - object-toworld matrix
 * */
template <class MatrixClass>
__global__ void transformation(DevObject* object, MatrixClass mTransform){

	object->transform(object, mTransform);
}

Polymorphic Mesh::getDevicePolymorphic(const Pattern& dev_pattern){
	return Polymorphic(
			  dev_intersect
			, dev_properties
			, dev_intersectBV
			, dev_transform
			, set_bv
			, dev_pattern
			);
}

Polymorphic Sphere::getDevicePolymorphic(const Pattern& dev_pattern)const{
  	return Polymorphic(
				  dev_sphere_intersect
				, dev_sphere_properties
				, dev_sphere_intersectBV
				, dev_sphere_transform
				, set_sphere_bv
				, dev_pattern
				);
  }
Polymorphic Plane::getDevicePolymorphic(const Pattern& dev_pattern)const{
  	return Polymorphic(
				  dev_plane_intersect
				, dev_plane_properties
				, dev_plane_intersectBV
				, dev_plane_transform
				, set_plane_bv
				, dev_pattern
				);
  }

Polymorphic Cube::getDevicePolymorphic(const Pattern& dev_pattern)const{
  	return Polymorphic(
				  dev_cube_intersect
				, dev_cube_properties
				, dev_cube_intersectBV
				, dev_cube_transform
				, set_cube_bv
				, dev_pattern
				);
  }

/*!
\addtogroup geometry_loading Loading geometry into the engine
\brief This module is designed to manage objects in both host and device memory.
Currently, it maintains the triangle mesh, the light source and the implicit instances storage.
@{
 */
template <class ObjectsType, class BoundingVolumeType>
class GeometryLoader{
	std::vector<ObjectsType *> meshes;
	std::vector<BoundingVolumeType *> boundaries;
	std::vector<DevObject *>	objects;
	std::vector<SquareMatrix4f> modelMatrices;
	std::vector<ILight> lights;
	std::vector<Appearence> appearences;
	textureMap textureNames;
	//allocate pointers for all boundary - vertex distances
	float * gpu_allDotProducts[ BoundingVolumeType::nNormals ];
/*!
* \brief Initializes a unordered map with string as key and pointer to device static function pointed
*  The string basically is a texture pattern name.
* */
	void setTextureNames(){
		textureNames = { {"none", &dev_none },
						{"strips", &dev_strips }
						,{"wave", &dev_wave }
						,{"grid",  &dev_grid }
						,{"checker",  &dev_checker } };
	}
/*!
* \brief Private function.
* It allocates auxiliary array to set bounding volume.
* */
	void allocateAllDotProduct(){
		try{
			size_t sizeOfAllDotProducts =
					(meshes.size() == 0 ) ? Cube::vertsNumber*sizeof(float) : meshes[0]->vertsNumber*sizeof(float);
			for(int i = 0; i < BoundingVolumeType::nNormals; ++i){
				allocateGPU(&gpu_allDotProducts[i],  sizeOfAllDotProducts);
			}
		}catch(const std::exception & e){
			throw CudaException( " allocateAllDotProduct -> " ) + e;
		}
	}
	/*!
	* \brief Private function.
	* It deallocates auxiliary array.
	* */
	void destroyAllDotProduct(){
		try{
			for(int i = 0; i < BoundingVolumeType::nNormals; ++i){
				checkError(  cudaFree( gpu_allDotProducts[i] ) );
			}
		}catch(const std::exception & e){
			throw CudaException( " cudaFree: " ) + e;
		}
	}
public:
	ObjectBox scene;

	GeometryLoader(const GeometryLoader& utilities):
				boundaries(utilities.boundaries)
			  , meshes(utilities.meshes)
			  , objects(utilities.objects)
			  , scene(utilities.scene){}
	/*!
	*\param[in] - filenames - storage for filenames with meshes. Each file must consist of the only one mesh.
	*\param[in] - model - object-to-world matrix storage, each for every mesh transformation.
	*\param[in] - lights - storage for lights of every implemented kind
	*\param[in] - appearences - object shading properties
	* */
	GeometryLoader(const std::vector<std::string>& filenames
		  , const std::vector<SquareMatrix4f>& model
		  , const std::vector<ILight>& lights
		  , const std::vector<Appearence> & appearences):
		 boundaries()
		,meshes( filenames.size() )
		,modelMatrices( model)
		,objects()
		,lights( lights )
	    ,appearences(appearences)
		,scene()
		{
			setTextureNames();
			 //load triangle meshes
			 loadMeshes(filenames);
			 //allocate helper memory
			 allocateAllDotProduct();
			 if( meshes.size() ){
				 //create wrapper objects
				 for(int i = 0; i < filenames.size(); ++i)
					 objects.push_back( new DevObject( meshes[i]->vertsNumber
							 	 	 	 	 	 	 , appearences[i] ) );
				 packMesh();
			 }
			 packScene();
		}
	/*!
	*\param[in] lights - storage for lights of every implemented kind
	* */
	GeometryLoader(const std::vector<ILight>& lights):
			 boundaries()
			,meshes( )
			,modelMatrices()
			,objects()
			,lights( lights )
		    ,appearences()
			,scene()
			{
				setTextureNames();
				runDemo();
				if(objects.size()){
					 //allocate helper memory
					allocateAllDotProduct();
					packScene();
				}
			}
	GeometryLoader(const std::vector<std::unique_ptr<HostObject>>& host_obj
			, const std::vector<ILight>& lights
			, const std::vector<Appearence> & appearences):
			 boundaries()
			,meshes( )
			,modelMatrices()
			,objects()
			,lights( lights )
		    ,appearences(appearences)
			,scene()
			{
				setTextureNames();
				for(int i = 0; i < host_obj.size() && i < appearences.size(); ++i){
					packProcedural(  host_obj[i]->getDevicePolymorphic( *textureNames[appearences[i].texture] )
									, appearences[i]
							        , *(host_obj[i].get() ) );
				}
				 //allocate helper memory
				allocateAllDotProduct();
				packScene();
			}
	/*!
	 * \detail 1) Applies transformation matrix to the mesh vertices.@n
	 * \detail 2) Recomputes bounding boundaries.
	 * \param[in] transform - a new transformation matrix
	 * \param[in] begin - the first point to be transformed
	 * \param[in] end - the point after the last one
	 * \return - ObjectBox wrapper of new scene considering transformations
	 * */
	  ObjectBox& operator()(const SquareMatrix4f& transform,const int& begin = 0, const int& end = 1){
		  //apply transformation matrix to the mesh vertices
		  assert(begin < end);
		  for(int i = begin; i < end; ++i){
			  transformation<SquareMatrix4f><<< gridDIM( objects[i]->vertsNumber, THREADS_ALONG_X)
						, THREADS_ALONG_X>>>(scene.data() + i, transform);
			  cudaDeviceSynchronize();
			  //recompute bounding boundaries
			  objects[i]->setBV( objects[i]
								 , boundaries[i]
								 , scene.boundingPlaneNormals
								 , gpu_allDotProducts);
		  }
		  return scene;
	  }
		/*!
		 * \detail Function recomputes a bounding boundaries of a particular object.
		 * \param[in] objectNumber
		 * \return ObjectBox wrapper of new scene considering transformations
		 * */
	  ObjectBox& operator()(int objectNumber){
		  objects[objectNumber]->setBV( objects[objectNumber]
		                                       , boundaries[objectNumber]
		                                       , scene.boundingPlaneNormals
		                                       , gpu_allDotProducts);
		  return scene;
	  }
		/*!
		 * \detail Function loads meshes from file.
		 * \param[in] filenames
		 * */
	  void loadMeshes(const std::vector<std::string>& filenames){
			int numberOfMeshes = filenames.size();
			if(numberOfMeshes){
				std::cout<<numberOfMeshes<<" meshes are loading..."<<std::endl;

				for(int i = 0; i < numberOfMeshes; ++i){
						std::cout<<"Processing..."<<std::endl;
						meshes[i] = loadFile(filenames[i], modelMatrices[i]);
				}
			}else{
				std::cout<<"No files have been loaded."<<std::endl;
			}
	  }
		/*!
		 * \detail Function loads implicit objects.
		 * */

	  void runDemo(){

		std::unique_ptr<Pipeline> pipeline(new MeshTransformation() );
		std::string materialType, texturing;
		Appearence appearence;
		std::cout<<"ATTENTION: A camera looks at the negative z axis by default."<<std::endl;
		std::string query;
		bool moveOn = true;
		while( moveOn && std::cout<<">" && std::getline(std::cin, query) )
		{
			if(query == "quit")
				moveOn = false;
			else if(query == "help"){
				std::cout<<"This is a rTTRacer help."
							"\nAvailable commands: "
							"\n background : set the background color."
							"\n cube : type to start cube object creation."
							"\n demo : launches demo-scene rendering."
							"\n help"
							"\n plane : type to start plane object creation."
							"\n sphere : type to start sphere object creation."
							<<std::endl;
			}else if(query == "bg"){
				input<Vec3f>(&scene.backgroundColor, "Input background color [ 0...1 0...1 0...1]:");
			}else if(query == "demo"){
		        //spiral
		       float spiralRadius = 7;
		       auto iterator = textureNames.begin();
		       std::unique_ptr<HostObject> object;

	           for(float i = 0; i < spiralRadius; i+=0.7, ++iterator){
	        	   srand (static_cast <unsigned> (time(0)));
	        	if( iterator == textureNames.end() ){
	        		iterator = textureNames.begin();
	        	}
	        	const Vec3f& _spiral = spiral( i - 8, i + 1, 2 );
	        	object.reset ( new Sphere( _spiral, 1, scene.boundingPlaneNormals ) );
	        	packProcedural( object->getDevicePolymorphic( *( iterator->second ) )
	        			, Appearence( Vec3f(random01(0.5),random01(0.5), random01(0.5)), 0.8f, 0.2f, 4.f, iterator->first.c_str() ).setTextureColor(Vec3f(random01(0.5), random01(0.5), random01(0.5) ) )
	        			, *object.get());

	        	if(i > spiralRadius/2){
					object.reset ( new Cube(_spiral*2, _spiral*2 + 2) );
		        	packProcedural(  object->getDevicePolymorphic( *textureNames["none"] )
		        			, Appearence (Vec3f(0, 0.5, 0.5), 0.8, 0.2, 4.f, "none" ).setTextureColor(Vec3f(0, 0.5, 0.5) )
		        			, *object.get());
	        	}
	           }
	        	Appearence plane(0.8);
	        	plane.setObjectColor(scene.backgroundColor);
	        	plane.setTextureColor(scene.backgroundColor);
				//plane
	        	object.reset ( new Plane( Vec3f(-20, 0, 20), Vec3f(20, 0, 20), Vec3f(20, 0, -25) ) );
	        	packProcedural(  object->getDevicePolymorphic( *textureNames[plane.texture] ), plane, *object.get());
		        moveOn = false;
				}
			else if(query == "sphere" || query == "cube" || query == "plane")
			{
				int numberOfProceduralMeshes = 0;
				input<int>(&numberOfProceduralMeshes, ( (std::string("Input a number of ") + query) + "s:").c_str());
				if(numberOfProceduralMeshes){

					for(int i = 0; i < numberOfProceduralMeshes;){
						float3 RGB[2];
						//set material
							input<std::string>(&materialType, "Input the material type [diffuse|phong|reflective]");
						if(materialType == "diffuse"){
							setColorAndTexture(RGB, texturing);
							appearence = Appearence(RGB[0], 0.9, 0.04, 50
									   						 , texturing.c_str()).setTextureColor( RGB[1] );

						}else if(materialType == "reflective"){
							appearence = Appearence(0.8);
							appearence.setTextureColor(scene.backgroundColor );
							appearence.setObjectColor(scene.backgroundColor );

						}else if(materialType == "phong"){
							setColorAndTexture(RGB, texturing);
							float3 diffuseSpecularExponent;// diffuse weight, specular weight, specular exponent
							std::cout<<"Input diffuse weight, specular weight, specular exponent separated with whitespaces or new line :"<<std::endl;
							std::cin>>diffuseSpecularExponent;
							appearence = Appearence(RGB[0], diffuseSpecularExponent.x, diffuseSpecularExponent.y, diffuseSpecularExponent.z , texturing.c_str() ).setTextureColor( RGB[1] );

						}else{
								std::cout<<std::string("Material: ") + materialType + " is not supported."<<std::endl;
								continue;
						}
						if(query == "sphere"){
								pipeline->reset();
								pipeline->translate( make_float3(rand()%15
																, rand()%15
																, rand()%15));
								Sphere tmp( pipeline->transform(), rand()%5, scene.boundingPlaneNormals );
								Polymorphic functions(
													 dev_sphere_intersect
													, dev_sphere_properties
													, dev_sphere_intersectBV
													, dev_sphere_transform
													, set_sphere_bv
													, *textureNames[appearence.texture]
														);
								packProcedural<Sphere>(tmp,functions, appearence);
						}else if(query == "plane"){
								printPlane();
								float4 points[3];
								inputPoints<3>(points);
								Plane tmp(points[0], points[1], points[2]);
								Polymorphic functions(
														  dev_plane_intersect
														, dev_plane_properties
														, dev_plane_intersectBV
														, dev_plane_transform
														, set_plane_bv
														, *textureNames[appearence.texture]
														);
										packProcedural<Plane>(tmp, functions, appearence);
						}else if(query == "cube"){
								float4 points[2];
								std::cout<<"Input the closest down left vertex and the far one."<<std::endl;
								inputPoints<2>(points);
								Cube cube( points[0], points[1] );
								Polymorphic functions(
													 dev_cube_intersect
													, dev_cube_properties
													, dev_cube_intersectBV
													, dev_cube_transform
													, set_cube_bv
													, *textureNames[appearence.texture]
													);
								packProcedural<Cube>( cube, functions, appearence );
										 }
						else{
								std::cout<<"This option is not available.";
							}
						++i;
				}
				std::cout<<"\nDone. You can continue or type quit / [Ecs] ."<<std::endl;

				std::cin.clear(); std::cin.ignore(INT_MAX,'\n');
				}else{
					continue;
				}
		}else{
				run_string(query, moveOn);
			}
		}
	}
/*!
* \brief Function copies meshes to device memory and initializes wrapper objects.
* */
	void packMesh(){
		for(int i = 0, j = 0; i < objects.size(); ++i)
		{
			try{
				Polymorphic functions(
			 	 	 	    dev_intersect
			 			 ,  dev_properties
			 			 ,  dev_intersectBV
			 			 ,  dev_transform
			 			 ,  set_bv
			 			, *textureNames[appearences[i].texture]);
				/*only if data hasn't been initialized at loading stage
				it could be an implicit object*/
				if( !objects[i]->data() ){
						boundaries.push_back( nullptr );
						objects[i]->init<ObjectsType, BoundingVolumeType>(
							functions
							, deleteMesh
							, meshes[j]
							, &boundaries[i]);
					++j;
				}
			}catch( const std::exception & e){
				throw CudaException(" Loader::packMesh -> ") + e;
			}
		}
	}
	/*!
	* \brief Function copies implicit objects to device memory and initializes wrapper objects.
	*\param[in] object - an instance of a particular object
	*\param[in] functions - instance with static function pointers
	*\param[in] appearences - object shading properties
	* */

	 template<class ProceduralObjectType>
	 void packProcedural( const ProceduralObjectType& object
				, const Polymorphic& functions
				, const Appearence& appearence){
		  try{
			  	objects.push_back(
			  			new DevObject(
			  					ProceduralObjectType::vertsNumber
			  					, appearence) );
			  	boundaries.push_back( nullptr );

			  	objects.back()->init< ProceduralObjectType, BoundingVolumeType>(
			  						  const_cast<Polymorphic &>(functions)
			  						, deleteMesh
			  						, &const_cast<ProceduralObjectType &>(object)
			  						, &boundaries.back());
		  }catch( const std::exception &e){
			  throw CudaException("packSphere ->") + e;
		  }
	  }

	 void packProcedural(  const Polymorphic& functions, const Appearence& appearence, const HostObject& object){
		  try{
			  	objects.push_back( new DevObject(object.getVertsNumber(), appearence) );
			  	boundaries.push_back( nullptr );

			  	objects.back()->init(
									 &const_cast<HostObject &>(object)
									, &boundaries.back()
			  						, const_cast<Polymorphic &>(functions)
			  						, deleteMesh
			  						);
		  }catch( const std::exception &e){
			  throw CudaException("packSphere ->") + e;
		  }
	  }

	/*!
	* \brief Function wrapper objects and light objects to the ObjectBox instance
	* */
	void packScene(){
		//allocate objects on device
		allocateGPU(&scene._array, objects.size()*sizeof(DevObject));
		for(int i = 0; i < objects.size(); ++i)
		{
			copyToDevice( &scene[i], objects[i], sizeof(DevObject));
		}
		scene._size = objects.size();
		//init lights and allocate loghts on device
		allocateGPU(&(scene.lights), lights.size()*sizeof(ILight));
		for(int i = 0; i < lights.size(); ++i)
		{
			if(lights[i].lightType == DISTANT_LIGHT){
				lights[i].init< DistantLight<Vec3f> >
					( 	dev_distant_reflected_color
						, dev_distant_illuminate );
			}else if(lights[i].lightType == POINT_LIGHT){
				lights[i].init< PointLight<Vec3f> >
					(  dev_point_reflected_color
					 , dev_point_illuminate );
			}
			copyToDevice( scene.lights + i, &lights[i], sizeof(ILight));
			scene.lightsNumber = lights.size();
		}
	}
	/*!
	* \return ObjectBox instance.
	* */
	ObjectBox & get(){ return scene; }

	~GeometryLoader(){
		scene.destroy();
		//free device memory with normals, vertices,
		//texture coordinates and vertex indexes in triangle structures
		for(auto m:meshes)
		{
			// delete meshes on host
			m->~Mesh();
		}
		for(auto o:objects)
		{
			//1. delete meshes on device
			o->destructor(o);
			//2. objects on host
			delete o;
		}
		//delete boundaries inside Loader this time
		for(auto b:boundaries){
			checkError( cudaFree(b) );
		}
		destroyAllDotProduct();
		cudaDeviceReset();
	}
};
/*! @} */
#endif /* GEOMETRYLOADER_CUH_ */
