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
#ifndef PARSER_CUH_
#define PARSER_CUH_
#include "objectBox.cuh"
VertexProperties getInt3(const char*& token);
static inline const char* parseSep(const char*& token);
static inline const char* parseSepOpt(const char*& token);
static inline float getFloat(const char*& token);
static inline Vec2f getVec2f(const char*& token);
static inline Vec3f getVec3f(const char*& token);
static inline bool isSep(const char c);

class CGParser{
protected:
	std::vector<VertexProperties> face;
	const SquareMatrix4f & model;
public:
	CGParser();
	CGParser(const SquareMatrix4f & model);
	//возможно, нужно сделать чисто виртуальной
	virtual Mesh* readFile(const char *filename);
	virtual ~CGParser();
};

class GeoParser: public CGParser{
public:
	GeoParser();
	GeoParser(const SquareMatrix4f& model);
	Mesh* readFile(const char *filename);
	~GeoParser();
};
class ObjParser: public CGParser{
	uint32_t numFaces; // количество граней
	std::vector<uint32_t> faceIndex; // из скольки вертексов состоит каждая грань
	std::vector<uint32_t> vertsIndex;// номера вертексов в verts которые составляют каждую из граней
	std::vector<Vec3f> verts;//v массив положений вертексов
	std::vector<Vec3f> normals;//vn массив нормалей к вертексам
	std::vector<Vec2f> st;//vt массив текстурных координат
	//VertexProperties face;
	//utilities function for storing indices in a right way if they defined 1-ordering like or inversed
	int fixV(int index) const;
	int fixVt(int index) const;
	int fixVn(int index) const;
	VertexProperties getInt3(const char*& token)const;
public:
	ObjParser();
	ObjParser(const SquareMatrix4f& model);
	Mesh* readFile(const char *filename);
	~ObjParser();
};

Mesh* loadFile( const std::string& filename, const SquareMatrix4f & model );

#endif /* PARSER_CUH_ */
