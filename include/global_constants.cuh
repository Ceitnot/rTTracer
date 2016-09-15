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
#ifndef __GLOBAL_CONSTANTS__
#define __GLOBAL_CONSTANTS__
#include <limits>
constexpr size_t STACK_LIMIT = 512; //in bytes
constexpr int THREADS_ALONG_X = 32;
constexpr int MAX_OBJECTS = 20;
constexpr int DIVISION_FACTOR  = 2;
constexpr float STEP  = 0.5f;
static constexpr float kInfinity = std::numeric_limits<int>::max();
typedef bool (*compare)(const float&,const float&  );
static constexpr size_t NORMALS = 7;
static constexpr size_t NORMAL_SET = 2;
///motion window offset
static constexpr size_t DELTA = 100;

constexpr int gridDIM( int dim, int threads )
{
    return ( dim + threads - 1 ) / threads;
}
///compute dimention of frame part for asynchronous computation
constexpr int partialDim( int dim )
{
    return ( dim + DIVISION_FACTOR  - 1) / DIVISION_FACTOR;
}

#endif
