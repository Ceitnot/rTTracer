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
#ifndef HELPER_CUH
#define HELPER_CUH
#include <exception>
#include <iostream>
class CudaException: public std::exception{
	std::string err_str;
public:
	CudaException(std::string& str):err_str(str){}
	CudaException(const char* str):err_str(str){}

	CudaException& operator+ (const std::exception & e){
		err_str += std::string( e.what() );
		return *this;
	}
	const char *what() const noexcept {return err_str.c_str();}
};

template <typename Type>
void checkAndNull(cudaError_t err, Type* ptr){
	if(err != cudaSuccess){
		std::cout<<cudaGetErrorString(err)<<" in "<<__FILE__<<" at line "<<__LINE__;
		std::cout<<std::endl;
	}else{
		ptr = NULL;
	}
}
template <typename Type>
void copyToDevice(Type* dst, const Type* src, size_t size){

	cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );
	if(err != cudaSuccess){
	//	std::cout<<cudaGetErrorString(err)<<" in "<<__FILE__<<" at line "<<__LINE__;
	//	std::cout<<std::endl;
		throw CudaException( cudaGetErrorString(err) );
	}

}
void checkError(cudaError_t err);
template <typename Type>
void allocateGPU(Type** pointer, size_t sizeOfMem){
	cudaError_t err = cudaMalloc( (void**)pointer, sizeOfMem );
	try{
	checkError(err);
	}catch(const std::exception & e){
		throw CudaException(" -> allocateGPU  - ") + e;
	}
}

template <typename Type>
void copyFromDevice(Type* dst, const Type* src, size_t size){
	cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );
	if(err != cudaSuccess){
		std::cout<<cudaGetErrorString(err)<<" in "<<__FILE__<<" at line "<<__LINE__;
		std::cout<<std::endl;
	}

}

template <typename Type>
__host__ __device__
void printMatrix(Type & matrix, uint8_t width, uint8_t height){
	for(int i = 0; i < width; ++i){
		for(int j = 0; j < height; ++j){
			printf("%f ", matrix[i][j] );
		}
		printf("\n");
	}
}
__host__ __device__
inline float modulo(const float &x)
{
    return x - floor(x);
}

std::istream& operator>>(std::istream& is, float3& dt);
std::istream& operator>>(std::istream& is, float4& dt);

template <class InpuData>
void input(InpuData* data, const char* message){
			std::cout<<message<<std::endl<<">";
			std::cin >> *data;
			std::cin.clear(); std::cin.ignore(INT_MAX,'\n');
}


template <class InpuData>
void commandInput(InpuData* data, const char** begin, const char** end){
	InpuData* _data = data;
	if(end > begin){
		for(const char** ptr = begin; ptr != end; ++ptr, ++_data){
			input(data, *ptr);
		}
	}
}



void run_string(const std::string& query, bool& moveOn);
__host__ __device__
float4 planeLowerRight(const float4& lowerLeftCorner, const float4& upperRightCorner);

template<size_t pointNumber>
void inputPoints(float4 (&points)[pointNumber] ){
	for(size_t i = 0; i < pointNumber; ++i){
		  std::cout<<"Input the "<<i<<" edge coordinates:";
		  std::cin>>points[i].x>>points[i].y>>points[i].z;
	}
}
void printPlane();

void setColorAndTexture(float3 (&color)[2], std::string& texturing);
float random01(float min = 0.0f);

#endif
