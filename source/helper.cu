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
#include "helper.cuh"
#include "algorithms.cuh"

/*!messages*/

/*!color and texture*/

const char* colorAndTexture[] = {
		"Enter the object color: r g b",
		":\nWhat type of procedural texturing would you like to use? [none|wave|strips|grid|checker]",
		"Enter the texture color: r g b",
		"Enter the material type [diffuse|phong|reflective]"
};

void checkError(cudaError_t err){

	if(err != cudaSuccess){
		throw CudaException( cudaGetErrorString(err) );
	}
}

void run_string(const std::string& query, bool& moveOn){
	const int ESC = 27;
	  for (const auto c : query)
	    {
	        if (c == ESC)
	        {
	            moveOn = false;
	            break;
	        }
	    }
	if(moveOn)
	{
		std::cout<<"There is no ["<< query <<"] command. Please, try again..."<<std::endl;
		std::cout<<">";
	}
}
void printPlane(){
	std::cout<<"3------2\n|      |\n0------1"<<std::endl;
}

std::istream& operator>>(std::istream& is, float3& dt)
{
    is >> dt.x >> dt.y >> dt.z;
    return is;
}

std::istream& operator>>(std::istream& is, float4& dt)
{
	is >> dt.x >> dt.y >> dt.z;
    dt.w = 0;
    return is;
}

void setColorAndTexture(float3 (&color)[2], std::string& texturing){
	commandInput<float3>(&color[0], colorAndTexture, colorAndTexture + 1);
	commandInput<std::string>(&texturing, colorAndTexture + 1, colorAndTexture + 2);
	if(texturing != "none"){
		commandInput<float3>(color + 1, colorAndTexture + 2, colorAndTexture + 3);
	}else{
		color[1] = color[0];
	}
}
float random01(float min){
	return  min + (rand()%10)/(float)10;

}
