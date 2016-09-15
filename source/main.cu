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
#include "bitmap.cuh"
#include "geometryLoader.cuh"
template <> const SquareMatrix4f SquareMatrix4f::Identity = SquareMatrix4f();
const char* greating="*************************\n"
					 "* rTTracer v0.9 Alpha   *\n"
					 "*************************";
#ifndef demo_1
const char* control_instructions="Control:\n\tMovement:\n\t\t"
		"- Move forward: W\n\t\t"
		"- Move backward: S\n\t\t"
		"- Move right: D\n\t\t"
		"- Move left: S\n\t"
		"Rotation:\n\t\t"
		"Press mouse and move it to the side you want turn to.\n\t"
		"Quit: press q or [ESC].";
#endif

int main(int argc, char **argv){
	std::cout<<greating<<std::endl;

#ifdef demo_1
	Vec3f boundingPlaneNormals[] = {
		    Vec3f(1, 0, 0),
		    Vec3f(0, 1, 0),
		    Vec3f(0, 0, 1),
		    Vec3f( sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f),
		    Vec3f(-sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f),
		    Vec3f(-sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f),
		    Vec3f( sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f) };
		//Create the camera settings
		Camera camera;
		//set up screen width and heght, field of view and scale
		camera.resolution(320, 240, 36.87, 0.5);
		//create a transformation pipeline. One can set an identity matrix to not make any changes.
		std::unique_ptr<Pipeline> pipeline( new MeshTransformation());
		pipeline->translate( make_float3(0,0,10));
		pipeline->scale( make_float3(1,1,1) );
		//Usually it's appropriate to perform the same transformation with both camera origin and camera direction.
		camera.modelDir = pipeline->transform() * defaultCameraMatrix.inverse();
		camera.modelOrig = camera.modelDir;


	    //set lights
	    std::vector<ILight> lights;
	    std::unique_ptr<Pipeline> light_pipeline( new MeshTransformation());
		SquareMatrix4f light_model(0.95292, 0.289503, 0.0901785, 0,
			 	      -0.0960954, 0.5704, -0.815727, 0,
			 	      -0.287593, 0.768656, 0.571365, 0,
			 		0, 0, 0, 1);
		//create 2 distant light sources
		light_pipeline->translate( make_float3(0,0,5));
		lights.push_back( ILight( light_model, Vec3f(1, 1, 1), 1.0f, DISTANT_LIGHT) );
		lights.push_back( ILight( light_pipeline->transform()*light_model, Vec3f(1, 1, 1), 1.0f, DISTANT_LIGHT) );

		//create an instance of the specular plane settings
		Appearence plane(0.8);
		Appearence sphere( Vec3f(1,0, 0), 0.8f, 0.2f, 4.f, "none" );
		plane.setObjectColor(Vec3f(0.5));
		plane.setTextureColor(Vec3f(0.5));
		sphere.setTextureColor(Vec3f(1, 0, 0 ) );
		// to get it into the engine one should pack it into stl vector
		std::vector<Appearence>  appearences;
		appearences.push_back(plane);
		appearences.push_back(sphere);
		//use interface HostObject to store instances of scene geometries
		std::vector<std::unique_ptr<HostObject>> host_obj;
		host_obj.push_back(std::unique_ptr<HostObject>(new Plane( Vec3f(-20, 0, 20), Vec3f(20, 0, 20), Vec3f(20, 0, -25) )));

		host_obj.push_back(std::unique_ptr<HostObject>(new Sphere( Vec3f(0, 10, -10), 3.0f, boundingPlaneNormals )   ));

		GeometryLoader<Mesh, Boundaries> loader(host_obj, lights, appearences);
#else

	    Camera camera;
	    camera.resolution(960, 540, 36.87, 0.5);
	    std::unique_ptr<Pipeline> pipeline( new MeshTransformation());
	    pipeline->translate( make_float3(0,0,10));
	    pipeline->scale( make_float3(1,1,1) );
	    camera.modelDir = pipeline->transform() * defaultCameraMatrix.inverse();
	    camera.modelOrig = camera.modelDir;

	    //set lights
	    std::vector<ILight> lights;
	    std::unique_ptr<Pipeline> light_pipeline( new MeshTransformation());

	    SquareMatrix4f l2w(0.95292, 0.289503, 0.0901785, 0,
	 		 -0.0960954, 0.5704, -0.815727, 0,
	 		 -0.287593, 0.768656, 0.571365, 0,
	 			 0, 0, 0, 1);
	    light_pipeline->translate( make_float3(0,0,5));
	    lights.push_back( ILight( l2w, Vec3f(1, 1, 1), 1.0f, DISTANT_LIGHT) );
	    lights.push_back( ILight( light_pipeline->transform()*l2w, Vec3f(1, 1, 1), 1.0f, DISTANT_LIGHT) );
	    //load objects with a simple command interface
	    GeometryLoader<Mesh, Boundaries> loader(lights);
#endif
	    BitmapMaker< GeometryLoader<Mesh, Boundaries> > bitmap(camera.width(), camera.height());
	    try{
	    	std::cout<<control_instructions<<std::endl;
	    	bitmap.display(camera, loader);
	    }catch(const std::exception& e){
	    	std::cout<<e.what()<<std::endl;
	    }
	    return 0;
}
