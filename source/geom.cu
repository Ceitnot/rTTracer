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

#include "geom.cuh"
#include "helper.cuh"

__host__ __device__
Color::Color()
:r(0), g(0), b(0){}
__host__ __device__
Color::Color(float red, float green, float blue)
			: r(red), g(green), b(blue){}
Color Color::operator*(const float& pattern) const{
	return Color(r*pattern, g*pattern, b*pattern);
}

__host__ __device__
Quaternion::Quaternion(float x, float y, float z, float w)
    	{
    		xyzw.x=x;
    		xyzw.y=y;
    		xyzw.z=z;
    		xyzw.w=w;
    	}
__host__ __device__
float Quaternion::length(){
    	return sqrtf(xyzw.x * xyzw.x + xyzw.y * xyzw.y + xyzw.z * xyzw.z + xyzw.w * xyzw.w);
    }
__host__ __device__
Quaternion Quaternion::Normalize()
{

    float len = length();

    if(!len) 
		return *this;

	float invLen = 1/len;
	xyzw.x *= invLen;
	xyzw.y *= invLen;
	xyzw.z *= invLen;
	xyzw.w *= invLen;
	return *this;

}
__host__ __device__
SquareMatrix4f Quaternion::toMatrix()
{
	*this = this->Normalize();

	float  xx = powf( xyzw.x, 2 );
	float  xy = xyzw.x * xyzw.y;
	float  xz = xyzw.x * xyzw.z;
	float  xw = xyzw.x * xyzw.w;

	float  yy = powf( xyzw.y, 2);
	float  yz = xyzw.y * xyzw.z;
	float  yw = xyzw.y * xyzw.w;
	float  zz = powf( xyzw.z, 2 );
	float  zw = xyzw.z * xyzw.w;
	float ww = powf( xyzw.w, 2 );

	return SquareMatrix4f( ww + xx - yy -zz,  2 * ( xy - zw ),       2 * ( xz + yw )     , 0
					, 2*(xy + zw),       ww - xx - yy - zz,     2 * ( yz - xw )     , 0
					, 2 * ( xz - yw ),      2 * ( yz + xw ),    ww - xx - yy + zz , 0
					, 0                    , 0                  , 0                 , 1);
}

void Camera::resolution(size_t _width,size_t _height, const float& _fov, const float& _scale){
	dimentions = make_uint4(_width, _height, partialDim(_width), partialDim(_height));
	displaySettings = make_float4(_fov, tan(deg2rad(_fov * _scale)), dimentions.x / (float)dimentions.y, 0);
}
__host__ __device__ const uint& Camera::width()const{
	return dimentions.x;
}
__host__ __device__ const uint& Camera::height()const{
	return dimentions.y;
}
__host__ __device__ const uint& Camera::partialWidth()const{
	return dimentions.z;
}
__host__ __device__ const uint& Camera::partialHeight()const{
	return dimentions.w;
}

__host__ __device__ const float& Camera::fov()const{
	return displaySettings.x;
}
__host__ __device__ const float& Camera::scale()const{
	return displaySettings.y;
}
__host__ __device__ const float& Camera::aspect()const{
	return displaySettings.z;
}
void Camera::offset(const int& x, const int& y){
	_offset = make_int2(x, y);
}
__host__ __device__ const int& Camera::xOffset()const{ return _offset.x; }
__host__ __device__ const int& Camera::yOffset()const{ return _offset.y; }

__host__ __device__
Quaternion Quaternion::conjugate(){

	return Quaternion( -xyzw.x, -xyzw.y, -xyzw.z, xyzw.w);
}
__host__ __device__
Quaternion operator*(const Quaternion& l, const Quaternion& r)
{

    const float w = (l.xyzw.w * r.xyzw.w) - (l.xyzw.x * r.xyzw.x)
    		- (l.xyzw.y * r.xyzw.y) - (l.xyzw.z * r.xyzw.z);

    const float x = (l.xyzw.x * r.xyzw.w) + (l.xyzw.w * r.xyzw.x)
    		+ (l.xyzw.y * r.xyzw.z) - (l.xyzw.z * r.xyzw.y);
    const float y = (l.xyzw.y * r.xyzw.w) + (l.xyzw.w * r.xyzw.y)
    		+ (l.xyzw.z * r.xyzw.x) - (l.xyzw.x * r.xyzw.z);
    const float z = (l.xyzw.z * r.xyzw.w) + (l.xyzw.w * r.xyzw.z)
    		+ (l.xyzw.x * r.xyzw.y) - (l.xyzw.y * r.xyzw.x);

    return Quaternion(x, y, z, w);
}
__host__ __device__
Quaternion operator*(const Quaternion& q, const Vec3f& v)
{
	const float w = - (q.xyzw.x * v.xyz.x)
			- (q.xyzw.y * v.xyz.y) - (q.xyzw.z * v.xyz.z);
	const float x =   (q.xyzw.w * v.xyz.x) + (q.xyzw.y * v.xyz.z)
			- (q.xyzw.z * v.xyz.y);
	const float y =   (q.xyzw.w * v.xyz.y) + (q.xyzw.z * v.xyz.x)
			- (q.xyzw.x * v.xyz.z);
	const float z =   (q.xyzw.w * v.xyz.z) + (q.xyzw.x * v.xyz.y)
			- (q.xyzw.y * v.xyz.x);

return Quaternion(x, y, z, w);
}

SquareMatrix4f scale(float x, float y, float z){
	return SquareMatrix4f( x, 0, 0, 0,
			  	  	  0, y, 0, 0,
			  	  	  0, 0, z, 0,
			  	  	  0, 0, 0, 1);
}
SquareMatrix4f translate(float x, float y, float z) {
	return SquareMatrix4f( 1, 0, 0, 0,
			  	  	  0, 1, 0, 0,
			  	  	  0, 0, 1, 0,
			  	  	  x, y, z, 1);

}
//axis needs to be normalized
SquareMatrix4f rotate(const Vec3f& axis, float degAngle){

    /* rotation quaternion
     * Q = [Vx*sin(a/2), Vy*sin(a/2), Vz*sin(a/2), cos(a/2)] (1)
     * or Q = [cos(a/2), U*sin(a/2)] (2) , where U - is normal vector axis, which
     * represents the angle of rotation. So, 'a' increases - sin gets bigger
      */
    const float SinHalfAngle = sinf(ToRadian(degAngle/2));
    /*
     * Half angle if half rotation in this method, so
     * because of (2) this formula  W = Q*V*Q^(-1) initially rotates one half
     * (let) Q' = Q*V (product is quaternian also)
     * and finally it rotates the second half Q'*Q^(-1).
     * We need two parts in order to rotate in hyper space and
     * then to translate it in 3D space.
     *  All because of the ordering of operations. */
    const float CosHalfAngle = cosf( ToRadian(degAngle/2) );
    std::cout<< "CosHalfAngle = "<< CosHalfAngle<<std::endl;
    const float rotationX = axis.xyz.x * SinHalfAngle;
    const float rotationY = axis.xyz.y * SinHalfAngle;
    const float rotationZ = axis.xyz.z * SinHalfAngle;
    const float rotationW = CosHalfAngle;

    Quaternion RotationQ(rotationX, rotationY, rotationZ, rotationW);

   return RotationQ.toMatrix();
}

Triangle::Triangle(){}
Triangle::Triangle(uint32_t v0, uint32_t v1, uint32_t v2){
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
	}
	__host__ __device__ uint32_t * Triangle::data(){
		return vertices;
	}
	__host__ __device__
	  const uint32_t& Triangle::operator[](size_t i) const {
		   return vertices[i];
	   }
	__host__ __device__
	uint32_t& Triangle::operator[](size_t i)  {
		   return const_cast<uint32_t&>
		   ( static_cast<const Triangle *>(this)->operator [](i) );
	   }

	__host__ __device__
	const Triangle& Triangle::operator = (const Triangle& tr) {
			for(int i = 0 ; i < 3; ++i)
				vertices[i] = tr.vertices[i];
		 return *this;
	 }
	__host__ __device__
	bool Triangle::hit( Ray& r, const Vec3f * verts, float& u, float&v)
	{
		Vec3f a = verts[0];
		Vec3f b = verts[1];
		Vec3f c = verts[2];

		Vec3f AB = b - a;
		Vec3f AC= c - a;
	    Vec3f pvec =
	    		r.dir.cross(AC);
	    float det = AB.dot(pvec);
	#ifdef CULLING
	    // if the determinant is negative the triangle is backfacing
	    // if the determinant is close to 0, the ray misses the triangle
	    if (det < kEpsilon) return false;
	#else
	    // ray and triangle are parallel if det is close to 0
	    if (fabs(det) < kEpsilon) return false;
	#endif
	    float invDet = 1 / det;

	    Vec3f tvec = r.orig - a;
	    u = tvec.dot(pvec) * invDet;
	    if (u < 0 || u > 1) return false;

	    Vec3f qvec = tvec.cross(AB);
	    v = r.dir.dot(qvec) * invDet;
	    if (v < 0 || u + v > 1) return false;

	    r.t = AC.dot(qvec) * invDet;

	   if( r.t > r.tMax || r.t < r.tMin) return false;
	    //save positive nearest point on the ray if calculated value is less then previous one
	    //or return false and don't save any
	    return ( r.t < r.tNearest) ? r.tNearest = r.t : false ;

		}
 __device__
Vec3f refract(const Vec3f &incident, const Vec3f &normal, const float &refarctionIndex)
{
	//1. find a cosine of an angle between an incedent ray and surface normal
    float cosi = clamp(-1, 1, incident.dot(normal)); // < 1 = |N| x |I| x cos(θ)
    float etai = 1 //the vacuum index
    , etat = refarctionIndex;
    Vec3f n = normal;
    if (cosi < 0) {
    	//the ray hit from the outside, so inverse the sign
    	cosi = -cosi;
    } else {
    	//the ray is outside of the second medium
    	algorithms::swap(etai, etat);
    	n= -normal;
    }
    float eta = etai / etat;
    float c2 = 1 - eta * eta * (1 - cosi * cosi);
    //also check if internal reflection happened
    //in this case there's no refraction
    return c2 < 0 ? 0 : eta * incident + (eta * cosi - sqrtf(c2)) * n;
}
/*!\fn void fresnel(const Vec3f &I, const Vec3f &N, const float &ior, float &kr)
 *\brief amount of reflected and refracted light computation
 *\detail computes the ratio between two waves: parallel and perpendicular polarised light
 * */
__host__ __device__
void fresnel(const Vec3f &incident, const Vec3f &normal, const float &refractionIndex, float &kr)
{
    float cosIncident = clamp(-1, 1, incident.dot(normal));
    float etaIncident = 1, etaTransmitted = refractionIndex;
    if (cosIncident > 0) { //if incident light inside the medium with greatest refraction index
    	algorithms::swap(etaIncident, etaTransmitted);
    }
    // Compute sinInternal using Snell's law
    float sinTransmitted = etaIncident / etaTransmitted * sqrtf( fmaxf(0.f, 1 - cosIncident * cosIncident) );
    // Total internal reflection
    if (sinTransmitted >= 1) {
        kr = 1;
    }
    else {
        float cosTransmitted = sqrtf(fmaxf(0.f, 1 - sinTransmitted * sinTransmitted));
        cosIncident = fabsf(cosIncident);
        float Rs = ( (etaTransmitted * cosIncident) - (etaIncident * cosTransmitted) ) / ((etaTransmitted * cosIncident) + (etaIncident * cosTransmitted));
        float Rp = ((etaIncident * cosIncident) - (etaTransmitted * cosTransmitted)) / ((etaIncident * cosIncident) + (etaTransmitted * cosTransmitted));
        kr = (Rs * Rs + Rp * Rp) / 2; // < take an average (ratio)
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}
__host__ __device__
float height(const Vec3f& top, const Vec3f& left, const Vec3f& right){
	float4 ABC = make_float4((top - left).length(), (right - top).length(), (left - right).length(), 0 );
	float p  = 0.5f *(ABC.x + ABC.y + ABC.z);
	return 2*sqrtf( p*(p - ABC.x)*(p - ABC.y)*(p - ABC.z) )/ABC.z;
}

__host__ __device__
Vec3f spiral(const float& t, const float& R, const float& S){
	return Vec3f( R * sinf(t), -S * t, R * cosf(t));
}

