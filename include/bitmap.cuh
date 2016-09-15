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
#ifndef BITMAP_CUH_
#define BITMAP_CUH_
#include <stdint.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include "rendering.cuh"
/*!
 \brief Cuda-OpenGL interoperability implementation.
 \addtogroup api_graphics Graphics APIs and animation
    @{
 */
// includes cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

const SquareMatrix4f defaultCameraMatrix (  0.707107,     -0.331295,  0.624695,  0
								     , 0,            0.883452,   0.468521,  0
								     , -0.707107,    -0.331295,  0.624695,  0
								     , -1.63871,     -5.747777, -40.400412, 1);
int chooseDevice();
template <class Functor>
class BitmapMaker{
	uint32_t sizeOfBitmap;
	unsigned char *pixels;
	Camera* camera;
	GLuint vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	Functor* functor;
	bool objectAnimation = true;
public:
	int x, y;
	BitmapMaker();
	BitmapMaker(int width,int height);
	~BitmapMaker();
	unsigned char* getPtr( void ) const ;
	long imageSize( void ) const ;

	void display(Camera& cam,
			Functor& resetData){
    	//initialize pointer to bitmap class for static callback functions.
    	BitmapMaker**   bitmap = getBitmapPtr();
    	*bitmap = this;
    	camera = &cam;
    	functor = &resetData;
    	glutIdleFunc(idle);
    	glutDisplayFunc(Draw);
    	glutKeyboardFunc(Key);
    	glutMotionFunc(motion);
        glutMainLoop();
    }
	unsigned char & operator[](uint32_t i);
	static BitmapMaker** getBitmapPtr( ) {
	      static BitmapMaker   *gBitmap;
	      return &gBitmap;
	    }
	bool initGL();

	static void Key(unsigned char key, int x, int y) {
		BitmapMaker*   bitmap = *(getBitmapPtr());
		std::unique_ptr<Pipeline>
			pipeline( new MeshTransformation() );
	    switch (key) {
	        case 27:
	        	glutDestroyWindow( glutGetWindow() );
	        	bitmap->deleteBuffer();
	        	exit(0);
	        case 'q':
	        	glutDestroyWindow( glutGetWindow() );
	        	bitmap->deleteBuffer();

	        	exit(0);
	        	//move camera forwards
	        case 'w':
	        	transformCamera(pipeline,  bitmap, make_float3( 0, 0, -STEP) );
	        	break;
	        	//move camera backwards
	        case 's':
	        	transformCamera(pipeline,  bitmap, make_float3( 0, 0, STEP) );
	        	break;
	        	//move camera right
	        case 'd':
	        	transformCamera(pipeline,  bitmap, make_float3( STEP, 0, 0) );
	        	break;
	        	//move camera left
	        case 'a':
	        	transformCamera(pipeline,  bitmap, make_float3( -STEP, 0, 0) );
	        	break;
	        case 'r':
	        	bitmap->camera->onKeyboard = true;
	        	bitmap->camera->modelOrig = defaultCameraMatrix.inverse();
	        	bitmap->camera->modelDir = bitmap->camera->modelOrig;
	        	break;
	        }
	}
	    // static method used for glut callbacks
	static void Draw( void ) {
		    BitmapMaker*   bitmap = *(getBitmapPtr());
		    glClearColor( 0.0, 0.0, 0.0, 1.0 );
		    glClear( GL_COLOR_BUFFER_BIT );
		    glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
		    glFlush();
		    glutSwapBuffers();
		        //glClear(GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
		    }
    void swap(BitmapMaker& bitmap);
    void createBuffer(unsigned int vbo_res_flags);
    void deleteBuffer();

    static void idle(){
    	BitmapMaker*   bitmap = *(getBitmapPtr());
    	if( bitmap->camera->onKeyboard || bitmap->camera->onMouse){
    		bitmap->camera->onKeyboard = false;
    		bitmap->camera->onMouse = false;

    		SquareMatrix4f transform;
			checkError(cudaGraphicsMapResources(1,
				&(bitmap->cuda_vbo_resource),
				0) );
			size_t numBytes;
			checkError(
					cudaGraphicsResourceGetMappedPointer(
							(void **)&(bitmap->pixels)
							, &numBytes
							, bitmap->cuda_vbo_resource) );

	if(bitmap->objectAnimation){
		GPU::render(*bitmap->camera
					, ( *(bitmap->functor) )(transform, 0, bitmap->functor->scene._size)
					, bitmap->pixels );
	//	bitmap->objectAnimation = false;
	}else{
		 GPU::render(*bitmap->camera
					, ( *(bitmap->functor) ).get()
					, bitmap->pixels );
	}
		checkError( cudaGraphicsUnmapResources(1, &bitmap->cuda_vbo_resource, 0) );
		glutPostRedisplay();
    	}
   }
    static void motion(int x, int y){
		BitmapMaker*   bitmap = *(getBitmapPtr());
		bitmap->camera->onMouse = true;

		std::unique_ptr<Pipeline>
		pipeline( new MeshTransformation() );
		if(x < DELTA){
					rotateCamera(pipeline, bitmap, Vec3f(0, 1, 0), -.5f);
				}else if(x >= bitmap->camera->width() - DELTA){
					rotateCamera(pipeline, bitmap, Vec3f(0, 1, 0), .5f);
				}
		if(y < DELTA){
					rotateCamera(pipeline, bitmap, Vec3f(1, 0, 0), -.5f);
				}else if(y >= bitmap->camera->height() - DELTA){
					rotateCamera(pipeline, bitmap, Vec3f(1, 0, 0), .5f);
				}
    }
    static void transformCamera(const std::unique_ptr<Pipeline>& pipeline
    						  , const BitmapMaker* bitmap
    						  , const Vec3f& data){
    	bitmap->camera->onKeyboard = true;
    	Vec3f axis(data);
    	bitmap->camera->modelDir.mulVecMat(axis, axis);
    	pipeline->translate( static_cast<float3>(axis) );
    	bitmap->camera->modelOrig = bitmap->camera->modelOrig
								* pipeline->transform();
    	bitmap->camera->modelDir = bitmap->camera->modelDir
								* pipeline->transform();
    }
    static void rotateCamera(const std::unique_ptr<Pipeline>& pipeline
    						  , const BitmapMaker* bitmap
    						  , const Vec3f& data
    						  , const float& delta){
    	Vec3f axis(data);
    	bitmap->camera->modelDir.mulVecMat(axis, axis);
		pipeline->rotate( static_cast<float3>(axis), delta );
		SquareMatrix4f transform( pipeline->transform() );
		bitmap->camera->modelDir = bitmap->camera->modelDir * transform;

    }
};
template <class Functor>
BitmapMaker<Functor>::BitmapMaker():
sizeOfBitmap(0)
, pixels(nullptr)
, camera(nullptr)
, vbo(0)
, cuda_vbo_resource(nullptr)
, functor(nullptr)
, x(0), y(0){}

template <class Functor>
BitmapMaker<Functor>::BitmapMaker(int width,int height):
										sizeOfBitmap(width*height*4)
										,x(width)
										,y(height)
										{
    	int device = chooseDevice();
    	if(device >= 0){
    		cudaGLSetGLDevice( device );
    		if( initGL()){
    			createBuffer( cudaGraphicsMapFlagsWriteDiscard );
    		}
    }
}


template <class Functor>
BitmapMaker<Functor>::~BitmapMaker(){ deleteBuffer(); }
template <class Functor>
unsigned char* BitmapMaker<Functor>::getPtr( void ) const   { return pixels; }
template <class Functor>
long BitmapMaker<Functor>::imageSize( void ) const { return sizeOfBitmap; }
template <class Functor>
unsigned char & BitmapMaker<Functor>::operator[](uint32_t i){ return pixels[i]; }

template <class Functor>
bool BitmapMaker<Functor>::initGL()
		{

	int argc = 0; char** argv = NULL;
	//init glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(x, y);
	glutInitWindowPosition(300, 100);
	glutCreateWindow( "rTTracer 0.9 Alpha" );
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glewInit();

	if (! glewIsSupported("GL_VERSION_2_0 "))
		{
		  fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		  fflush(stderr);
		  return false;
		 }
		 return true;
		}

template<class Functor>
 void BitmapMaker<Functor>::swap(BitmapMaker& bitmap){
	  std::swap(bitmap.pixels, pixels);
	  std::swap(bitmap.sizeOfBitmap, sizeOfBitmap);
	  std::swap(bitmap.x, x);
	  std::swap(bitmap.y, y);
 }

template<class Functor>
 void BitmapMaker<Functor>::createBuffer(unsigned int vbo_res_flags){
	    // create buffer object
	    glGenBuffers(1, &vbo);
	    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	    // initialize buffer object
	    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeOfBitmap, nullptr, GL_DYNAMIC_DRAW);
	    // register this buffer object with CUDA
	    checkError(
	    		cudaGraphicsGLRegisterBuffer(
	    				&cuda_vbo_resource, vbo
	    				, vbo_res_flags));
	    cudaGraphicsResourceSetMapFlags	( cuda_vbo_resource
	    		, cudaGraphicsMapFlagsWriteDiscard);
 }

template<class Functor>
 void BitmapMaker<Functor>::deleteBuffer(){
	    // unregister this buffer object with CUDA
	    checkError(cudaGraphicsUnregisterResource(cuda_vbo_resource) );
	    glBindBuffer(1, vbo);
	    glDeleteBuffers(1, &vbo);
	    vbo = 0;
 }

/*! @} */
#endif /* BITMAP_CUH_ */
