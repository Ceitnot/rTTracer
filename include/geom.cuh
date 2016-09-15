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
#ifndef GEOM_CUH_
#define GEOM_CUH_
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <vector>
#include <assert.h>
#include "global_constants.cuh"
#include "algorithms.cuh"
#include "helper.cuh"
/*! \defgroup linear_algebra Linear algebra
    \ingroup wrapping_and_description
    @{
*/
constexpr float kEpsilon = 1e-8;
__host__ __device__
inline float clamp(const float &lo, const float &hi, const float &v)
{ return fmax(lo, fmin(hi, v)); }

constexpr double ToRadian(float x) {
	return x * M_PI/ 180.0f ;
}
__host__ __device__
inline
float deg2rad(const float &deg) { return deg * M_PI / 180; }

class Color{
public:
	float r, g, b;
	__host__ __device__
	Color();
	__host__ __device__
	Color(float red, float green, float blue);
	__host__ __device__
	Color operator*(const float& pattern) const;
};

/****/
/**Vectors and Matrices**/
class Vec3 ;
//class DevObject;

class Vec2
{
public:
	float2 xy;

    // 3 most basic ways of initializing a vector
	__host__ __device__ Vec2():
							xy( make_float2(0, 0) ){}
	__host__ __device__ Vec2(const float &xx):
							xy( make_float2(xx, xx) ){}

	__host__ __device__ Vec2(float xx, float yy):
							xy( make_float2(xx, yy) ){}
	__host__ __device__ Vec2(const Vec2& v):
							xy( make_float2(v.xy.x, v.xy.y) ){}
    __host__ __device__ Vec2 operator + (const Vec2 &v) const
   { return Vec2(xy.x + v.xy.x, v.xy.y + xy.y); }
    Vec2 operator / (const float &r) const
    { return Vec2(xy.x / r, xy.y / r); }
    __host__ __device__ Vec2 operator * (const float &r) const
   { return Vec2(xy.x * r, xy.y * r); }

    __host__ __device__
    Vec2& operator /= (const float &r)
    { xy.x /= r, xy.y /= r; return *this; }
    __host__ __device__
    Vec2& operator *= (const float &r)
    { xy.x *= r, xy.y *= r; return *this; }
    friend std::ostream& operator << (std::ostream &s, const Vec2 &v)
    {
        return s << '[' << v.xy.x << ' ' << v.xy.y << ']';
    }
    __host__ __device__
    friend Vec2 operator * (const float &r, const Vec2 &v)
    { return Vec2(v.xy.x * r, v.xy.y * r); }
    __host__ __device__
    friend Vec2 operator / (const float &r, const Vec2 &v)
    { return Vec2(r/v.xy.x, r/v.xy.y); }
    __host__ __device__
    void swap(Vec2& vec){
    	float2 tmp = vec.xy;
    	vec.xy = xy;
    	xy = tmp;
    }

};

class Vec3
{

public:
	float4 xyz;
	__host__ __device__ Vec3():xyz( make_float4( 0, 0, 0, 0 ) ){}
	__host__ __device__ Vec3(float xx): xyz( make_float4(xx, xx, xx, xx) ){}
	__host__ __device__ Vec3(float xx, float yy, float zz): xyz( make_float4(xx, yy, zz, 0) ){}
	__host__ __device__ Vec3(const Vec3& vec):
									xyz( make_float4(vec.xyz.x,
													 vec.xyz.y,
													 vec.xyz.z,
													 0) ){}
	__host__ __device__ Vec3(const float3& vec):xyz(make_float4(vec.x, vec.y, vec.z, 0)){}
	__host__ __device__ Vec3(const float4& vec):xyz(make_float4(vec.x, vec.y, vec.z, 0)){}
    //operators
    __host__ __device__ float& operator [] (uint8_t i) {

		 float *result = NULL;
		 if(i > 2) {
			 i %= 2;
		 }
		 switch(i){
		 case 0: result = &xyz.x; break;
		 case 1: result = &xyz.y; break;
		 case 2: result = &xyz.z; break;
		 }
		 return *result;
    }

    __host__ __device__ const float& operator [] (uint8_t i) const {

   	 return reinterpret_cast<const float&>( (*this)[i] ); }

    inline __host__ __device__ Vec3 operator + (const Vec3 &v) const
   { return Vec3(xyz.x + v.xyz.x, xyz.y + v.xyz.y, xyz.z + v.xyz.z); }
    inline __host__ __device__ Vec3& operator += (const Vec3 &v)
   {
    	*this = *this + v;
    	return *this;
   }
    inline __host__ __device__ Vec3 operator - (const Vec3 &v) const
   { return Vec3(xyz.x - v.xyz.x, xyz.y - v.xyz.y, xyz.z - v.xyz.z); }
    inline __host__ __device__ Vec3 operator - () const
   { return Vec3(-xyz.x, -xyz.y, -xyz.z); }
    inline __host__ __device__ Vec3 operator * (const float &r) const
		{
    	return Vec3(xyz.x * r, xyz.y * r, xyz.z * r); }
    inline __host__ __device__ Vec3 operator *= (const float &r) const
   { return (*this) * r; }
    inline __host__ __device__ Vec3 operator * (const Vec3 &v) const
    { return Vec3(xyz.x * v.xyz.x, xyz.y * v.xyz.y, xyz.z * v.xyz.z); }

    inline __host__ __device__ Vec3 operator / (const float &r) const
   {
   	 float denominator = 1.0f/(float)r;
   	 return Vec3(xyz.x * denominator, xyz.y * denominator, xyz.z * denominator);
   }

    inline __host__ __device__ Vec3& operator= (const Vec3 &vec)
   {
    	xyz = vec.xyz;
    	return *this;
   	}
    inline __host__ __device__ operator float3()const{
    	return make_float3(xyz.x, xyz.y, xyz.z);
    }
    inline  __host__ __device__
    friend Vec3 operator * (const float &r, const Vec3 &v)
    { return Vec3(v.xyz.x * r, v.xyz.y * r, v.xyz.z * r); }
    inline  __host__ __device__
    friend Vec3 operator / (const float &r, const Vec3 &v)
    { return Vec3(r / v.xyz.x, r / v.xyz.y, r / v.xyz.z); }


	__host__ __device__ float length()const{
    	return sqrtf( xyz.x*xyz.x   +  xyz.y*xyz.y  +  xyz.z*xyz.z);
    }
     __host__ __device__ Vec3& normalize(){

    	float len = length();

    	if(!len) return *this;

    	else{
    		float invLen = 1/len;
    		xyz.x *= invLen, xyz.y *= invLen, xyz.z *= invLen;
    		return *this;
    	}
    }
     __host__ __device__  Vec3 cross(const Vec3 &v) const
    {
        return Vec3(

        		xyz.y * v.xyz.z - xyz.z * v.xyz.y,
        		xyz.z * v.xyz.x - xyz.x * v.xyz.z,
        		xyz.x * v.xyz.y - xyz.y * v.xyz.x);
    }
     __host__ __device__  float dot(const Vec3 &v) const
    {
        return xyz.x*v.xyz.x + xyz.y*v.xyz.y + xyz.z*v.xyz.z;
    }
     inline  friend std::ostream& operator << (std::ostream &s, const Vec3 &v)
    {
        return s << '(' << v.xyz.x << ' ' << v.xyz.y << ' ' << v.xyz.z << ')';
    }
     inline  friend std::istream& operator >> (std::istream &is, Vec3 &v)
    {
        return is >> v.xyz;
    }
     __host__ __device__
     void swap(Vec3& vec ){
    	 float4 tmp = make_float4( vec.xyz.x
    			 	 	 	 	 , vec.xyz.y
    			 	 	 	 	 , vec.xyz.z
    			 	 	 	 	 , 0);
    	 vec.xyz = xyz;
    	 xyz = tmp;
     }
};

template<typename T>
class SquareMatrix4
{
public:

    T x[4][4] = { {1,0,0,0}
    			 ,{0,1,0,0}
    			 ,{0,0,1,0}
    			 ,{0,0,0,1} };

    static const SquareMatrix4 Identity;

    inline __host__ __device__
    SquareMatrix4() {}
    inline __host__ __device__
    SquareMatrix4 (T a, T b, T c,T d,
    		  T e, T f, T g, T h,
    		  T i, T j, T k, T l,
    		  T m, T n, T o, T p)
    {
		x[0][0] = a; x[0][1] = b;  x[0][2] = c; x[0][3] = d;
		x[1][0] = e; x[1][1] = f;  x[1][2] = g; x[1][3] = h;
		x[2][0] = i; x[2][1] = j;  x[2][2] = k; x[2][3] = l;
		x[3][0] = m; x[3][1] = n;  x[3][2] = o; x[3][3] = p;
    }

    inline __host__ __device__
    SquareMatrix4(const SquareMatrix4& matrix){
    	for(int i = 0; i < 4; ++i){
    		for(int j = 0; j < 4; ++j){
    			x[i][j] = matrix.x[i][j];
    		}
    	}
    }
    __host__ __device__
    const T* operator [] (uint8_t i) const { return x[i]; }
    __host__ __device__
    T* operator [] (uint8_t i) {
    	return const_cast<T*>
    	( static_cast<const SquareMatrix4 *>(this)->operator [](i) );
    }
    __host__ __device__
    inline  SquareMatrix4 operator * (const SquareMatrix4& v) const
    {
        SquareMatrix4 tmp;
        multiply (*this, v, tmp);
        return tmp;
    }
    __host__ __device__
    inline  SquareMatrix4& operator *= (const SquareMatrix4& v) const
    {
    	*this = (*this) * v;
        return *this;
    }
    __host__ __device__
    inline  SquareMatrix4& operator = (const SquareMatrix4& matrix)
    {
    	for(int i = 0; i < 4; ++i){
    		for(int j = 0; j < 4; ++j){
    			x[i][j] = matrix.x[i][j];
    		}
    	}
        return *this;
    }
    __host__ __device__
    static void multiply(const SquareMatrix4<T> &a, const SquareMatrix4& b, SquareMatrix4 &c)
    {
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] +
                    a[i][2] * b[2][j] + a[i][3] * b[3][j];
            }
        }
    }
    /*!  return a transposed copy of the current matrix as a new matrix
    */
    __host__ __device__
    inline  SquareMatrix4 transposed() const
    {
        SquareMatrix4 transpose;
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                transpose[i][j] = x[j][i];
            }
        }
        return transpose;
    }
    /*! \brief transpose itself
     */
    __host__ __device__
    inline  SquareMatrix4& transpose ()
    {
        *this = transposed();
        return *this;
    }
/*!
 * \brief multiply point by matrix
 * \param [in] src - vector multyplied by a matrix
 * \param [out] dst - output vector*/

    inline  __host__ __device__
    void mulPointMat(const Vec3 &src, Vec3 &dst) const
    {
       float a, b, c;
        a = src.xyz.x * x[0][0] + src.xyz.y * x[1][0] + src.xyz.z * x[2][0] + x[3][0];
        b = src.xyz.x * x[0][1] + src.xyz.y * x[1][1] + src.xyz.z * x[2][1] + x[3][1];
        c = src.xyz.x * x[0][2] + src.xyz.y * x[1][2] + src.xyz.z * x[2][2] + x[3][2];

        dst.xyz.x = a;
        dst.xyz.y = b;
        dst.xyz.z = c;
    }
    /*!
     * \brief multiply vector by matrix
     * \param [in] src - vector multyplied by a matrix
     * \param [out] dst - output vector*/
    __host__ __device__
    void mulVecMat(const Vec3 &src, Vec3 &dst) const
    {
        float a, b, c;

        a = src.xyz.x * x[0][0] + src.xyz.y * x[1][0] + src.xyz.z * x[2][0];
        b = src.xyz.x * x[0][1] + src.xyz.y * x[1][1] + src.xyz.z * x[2][1];
        c = src.xyz.x * x[0][2] + src.xyz.y * x[1][2] + src.xyz.z * x[2][2];

        dst.xyz.x = a;
        dst.xyz.y = b;
        dst.xyz.z = c;
    }
__host__ __device__
    SquareMatrix4 inverse() const
    {
        int i, j, k;
        SquareMatrix4 s;
        SquareMatrix4 t (*this);

        // Forward elimination
        for (i = 0; i < 3 ; i++) {
            int pivot = i;

            T pivotsize = t[i][i];

            if (pivotsize < 0)
                pivotsize = -pivotsize;

                for (j = i + 1; j < 4; j++) {
                    T tmp = t[j][i];

                    if (tmp < 0)
                        tmp = -tmp;

                        if (tmp > pivotsize) {
                            pivot = j;
                            pivotsize = tmp;
                        }
                }

            if (pivotsize == 0) {
                // Cannot invert singular matrix
                return SquareMatrix4();
            }

            if (pivot != i) {
                for (j = 0; j < 4; j++) {
                    T tmp;

                    tmp = t[i][j];
                    t[i][j] = t[pivot][j];
                    t[pivot][j] = tmp;

                    tmp = s[i][j];
                    s[i][j] = s[pivot][j];
                    s[pivot][j] = tmp;
                }
            }

            for (j = i + 1; j < 4; j++) {
                T f = t[j][i] / t[i][i];

                for (k = 0; k < 4; k++) {
                    t[j][k] -= f * t[i][k];
                    s[j][k] -= f * s[i][k];
                }
            }
        }

        // Backward substitution
        for (i = 3; i >= 0; --i) {
            T f;

            if ((f = t[i][i]) == 0) {
                // Cannot invert singular matrix
                return SquareMatrix4();
            }

            for (j = 0; j < 4; j++) {
                t[i][j] /= f;
                s[i][j] /= f;
            }

            for (j = 0; j < i; j++) {
                f = t[j][i];

                for (k = 0; k < 4; k++) {
                    t[j][k] -= f * t[i][k];
                    s[j][k] -= f * s[i][k];
                }
            }
        }

        return s;
    }

    /*!
     *  \brief set current matrix to its inverse
     */
    const SquareMatrix4<T>& invert()
    {
        *this = inverse();
        return *this;
    }

};
// constructor allows for implicit conversion

typedef Vec3 Vec3f;
typedef Vec2 Vec2f;
typedef SquareMatrix4<float> SquareMatrix4f;


enum RayType { PRIMARY_RAY, SHADOW_RAY};

class Ray{
public:
	__host__ __device__
    Ray(): orig(0.f)
			, dir(0, 0, -1)
			, inverseDir(1/dir)
			, tMin(0.1)
			, tMax(1000)
			, t(-kInfinity)
			, tNearest(kInfinity)
			, rayType(PRIMARY_RAY)
{}
	__host__ __device__  Ray(const Vec3f &o
							, const Vec3f &d, const RayType& rayType = PRIMARY_RAY)
											: orig(o)
											, dir(d)
											, inverseDir( 1/dir )
											, tMin(0.1)
											, tMax(1000)
											, t(-kInfinity)
											, tNearest(kInfinity)
											, rayType(rayType)
    										{}
    __host__ __device__
    Ray(const Ray& ray):
    			 orig( ray.orig )
               , dir( ray.dir )
               , inverseDir(ray.inverseDir)
    		   , tMin (ray.tMin)
               , tMax( ray.tMax )
	   	   	   , t ( ray.t )
			   , tNearest( ray.tNearest )
    		   , rayType(ray.rayType)
               {}
    __host__ __device__
    Ray & operator=(const Ray& ray){
    	Ray tmp(ray);
    	swap(tmp);
        return *this;
    }
    __host__ __device__
    void swap(Ray& ray){
    	ray.dir.swap(dir);
    	ray.inverseDir.swap(inverseDir);
    	ray.orig.swap(orig);
    	algorithms::swap<RayType>(ray.rayType, rayType);
    	algorithms::swap<float>(ray.t, t);
    	algorithms::swap<float>(ray.tMax, tMax);
    	algorithms::swap<float>(ray.tMin, tMin);
    	algorithms::swap<float>(ray.tNearest, tNearest);
    }
    __host__ __device__
    float4 direction(){
    	return make_float4(dir.xyz.x, dir.xyz.y, dir.xyz.z,0);
    }
    Vec3f orig, dir, inverseDir;
    float tMin, tMax
    , t			//current intersect distance along the ray
    , tNearest;//minimal intersect distance along the ray
    RayType rayType;

};
/*!
 * A camera class. Rays are transmitted through the origin along the direction.
 * The position of the camera is stored in model matrix.
 */
struct Camera {
    SquareMatrix4f modelDir, modelOrig;
	bool onKeyboard = false;/// if key is pressed
	bool onMouse = false;

	Vec3f up = Vec3f(0, 1, 0);
	Vec3f right = Vec3f(1, 0, 0);

private:
	/*!
	actual display dimensions divided by
	DIVISION_FACTOR in order to get it computet in different cuda streams
	x - partial width
	y - partial height
	z - full width
	w - full height
	*/
	uint4 dimentions;
	/*!
	 x - fov, y - scale, z - aspect ratio, w - zero*/
	float4 displaySettings;

	int2 _offset;/// for asyncronous kernel launch
public:
	void resolution(size_t _width,size_t _height, const float& _fov, const float& _scale);
	__host__ __device__ const uint& width()const;
	__host__ __device__ const uint& height()const;
	__host__ __device__ const uint& partialWidth()const;
	__host__ __device__ const uint& partialHeight()const;
	__host__ __device__ const float& fov()const;
	__host__ __device__ const float& aspect()const;
	__host__ __device__ const float& scale()const;
	void offset(const int& x, const int& y);
	__host__ __device__ const int& xOffset()const;
	__host__ __device__ const int& yOffset()const;
};

class Quaternion
{
public:
    float4 xyzw;
    __host__ __device__
    Quaternion(float x, float y, float z, float w);
    __host__ __device__
    float length();
    __host__ __device__
    Quaternion Normalize();
    __host__ __device__
    SquareMatrix4f toMatrix();
	/*!
	 \f$ Q = (V, w)\f$  or \f$ Q = (sin(a/2), U*sin(a/2) ) \f$,
	 to inverse we should get \f$sin\f$ negative because \f$cos\f$ will not has any effect
	 either \f$V\f$ or \f$w\f$ should be negative to inverse rotation*/
    __host__ __device__
    Quaternion conjugate();
 };

__host__ __device__
Quaternion operator*(const Quaternion& l, const Quaternion& r);
__host__ __device__
Quaternion operator*(const Quaternion& q, const Vec3f& v);

SquareMatrix4f scale(float x, float y, float z);
SquareMatrix4f translate(float x, float y, float z);
/*!\warning axis needs to be normalized*/
SquareMatrix4f rotate(const Vec3f& axis, float degAngle);

class Triangle{
	uint32_t vertices[3];

public:

	Triangle();
	Triangle(uint32_t v0, uint32_t v1, uint32_t v2);
	__host__ __device__ uint32_t * data();
	__host__ __device__
	const uint32_t& operator[](size_t i)const;

	__host__ __device__
	uint32_t& operator[](size_t i);

	__host__ __device__
	const Triangle& operator = (const Triangle& tr);

	__host__ __device__
	bool hit( Ray& r, const Vec3f * verts, float& u, float&v);
};
namespace GPU{
	struct RenderingContext{
		__host__ __device__
		RenderingContext():
			 triangleIndex(0)
			,textureCoordinates(0)
			,hitNormal(0)
		    ,lightDir(0,0,-1)
		    ,lightIntensity(1)
			, index(0)
		{}
		__host__ __device__
		RenderingContext(const RenderingContext& rctx)
											: triangleIndex( rctx.triangleIndex )
											, textureCoordinates ( rctx.textureCoordinates )
											, hitNormal( rctx.hitNormal )
											, lightDir(rctx.lightDir)
											, lightIntensity(rctx.lightIntensity)
											, index(0){}
		__host__ __device__
		RenderingContext& operator = (const RenderingContext& rctx)
		{
			GPU::RenderingContext(rctx).swap(*this);
			return *this;
		}
		__host__ __device__
		void swap(RenderingContext& rctx){
			rctx.hitNormal.swap(hitNormal);
			rctx.lightDir.swap(lightDir);
			rctx.lightIntensity.swap(lightIntensity);
			rctx.textureCoordinates.swap(textureCoordinates);
			algorithms::swap(rctx.triangleIndex, triangleIndex);
		}
		//objects
		uint32_t triangleIndex;
		Vec2f textureCoordinates;
		Vec3f hitNormal;
		int index; // if object is complex
		const float bias = 0.001f;
		//lights
		Vec3f lightDir, lightIntensity;
		const float maxDepth = 5;
	};
	//should be overloaded while adding another types of bounding volumes
}
/*!
 * \brief auxiliary class for .obj files parsing.
 * */
struct VertexProperties {

    int3 vert;
    VertexProperties() {};
    VertexProperties(int v) : vert( make_int3(v,v,v) ) {};
    VertexProperties(int v, int vt, int vn) : vert( make_int3(v, vt, vn) ) {};
};

/*! \file
 * 	\fn Vec3f reflect(const Vec3f &incident, const Vec3f &normal)
 * \brief Reflection computation
 * 	\f$ R = I − cos(θ)∗N − cos(θ)∗N \f$ (1) the formula for finding the reflection direction
 * \param [in] incident - the directional vector of an incident ray
 * \param [in] normal - a surface normal
 * */
//__host__ __device__
//Vec3f reflect(const Vec3f &incident, const Vec3f &normal);
/*! \fn Vec3f refract(const Vec3f &incident, const Vec3f &normal, const float &refarctionIndex)
 * \brief Refraction computation
 * \details
 * \f$ T=ηI+(ηc_1−c_2)N\f$ ,
 * \details where  \f$η = η_1/η_2 (η\f$ is a refraction coefficient in the medium),
 * \details
 * \f$c_1 = \cos(\theta_1) = N \cdot I,\\ \f$
 * \f$ c_2 = \sqrt{1 - \left( \frac{\eta_1}{\eta_2} \right) ^2 \sin^2(\theta_1)} \rightarrow \sqrt{1 - \left( \frac{\eta_1}{\eta_2} \right) ^2 (1 - \cos^2(\theta_1))}\f$
 * \param [in] incident - the directional vector of an incident ray
 * \param [in] normal - a surface normal
 * \param [in] refarctionIndex - refraction coefficient(of an object) is simply a vacuum speed of light divided by the speed of specific medium
 * */
__host__ __device__
Vec3f refract(const Vec3f &incident, const Vec3f &normal, const float &refarctionIndex);
__host__ __device__
void fresnel(const Vec3f &incident, const Vec3f &normal, const float &refractionIndex, float &kr);

__host__ __device__
float height(const Vec3f& top, const Vec3f& left, const Vec3f& right);

//__host__ __device__
//float2 texCoordinates(const Plane& plane, const Vec3f& hitPoint);
/*!
x(t) = R * sin(t)
y(t) = R * cos(t)
z(t) = S * t
Let t - parameter, R - spiral radius, S - скорость шага (вытянутость вдоль стержня)
 */
__host__ __device__
Vec3f spiral(const float& t, const float& R, const float& S);

void setColorAndTexture(Vec3f& objectColor, Vec3f& textureColor, std::string& texturing);
/*!
    @}
*/
#endif /* GEOM_CUH_ */
