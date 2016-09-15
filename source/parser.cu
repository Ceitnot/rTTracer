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
#include "parser.cuh"
#include <stdexcept>
#include <string>
#include <cstring>
CGParser::CGParser():face()
		,model( SquareMatrix4f() ){}
Mesh* CGParser::readFile(const char *filename){
	return nullptr;
}

CGParser::CGParser(const SquareMatrix4f & model):face()
					,model( model ){}
CGParser::~CGParser(){}
GeoParser::GeoParser():CGParser(){}
GeoParser::GeoParser(const SquareMatrix4f& model)
						:CGParser(model){}
Mesh* GeoParser::readFile(const char *filename){
	std::ifstream ifs;
	    try {
	        ifs.open(filename);
	        if (ifs.fail()) throw;

	        std::stringstream ss;
	        ss << ifs.rdbuf();
	        uint32_t numFaces;
	        ss >> numFaces;
	        std::vector<uint32_t> faceIndex(numFaces);
	        uint32_t vertsIndexArraySize = 0;
	        for (uint32_t i = 0; i < numFaces; ++i) {
	             ss >> faceIndex[i];
	            vertsIndexArraySize += faceIndex[i];
	        }

	       std::vector<uint32_t> vertsIndex(vertsIndexArraySize);
	        uint32_t vertsArraySize = 0;

	        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
	        	ss >> vertsIndex[i];
	            if (vertsIndex[i] > vertsArraySize) vertsArraySize = vertsIndex[i];
	        }
	        vertsArraySize += 1;
	        std::vector<Vec3f> verts(vertsArraySize);
	        for (uint32_t i = 0; i < vertsArraySize; ++i) {
	            ss >> verts[i].xyz.x >> verts[i].xyz.y >> verts[i].xyz.z;
	        }
	        std::vector<Vec3f> normals(vertsIndexArraySize);
	        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
	            ss >> normals[i].xyz.x >> normals[i].xyz.y >> normals[i].xyz.z;
	        }
	        std::vector<Vec2f> st(vertsIndexArraySize);
	        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
	            ss >> st[i].xy.x >> st[i].xy.y;
	        }

	        return new Mesh(numFaces, faceIndex, vertsIndex, verts, normals, st, model);
	    }
	    catch (...) {
	        ifs.close();
	    }
	    return nullptr;
}
GeoParser::~GeoParser(){}

ObjParser::ObjParser(): CGParser()
						,numFaces(0)
						,faceIndex()
						,vertsIndex()
						,verts()
						,normals()
						,st()
						{}
ObjParser::ObjParser(const SquareMatrix4f& model)
							: CGParser(model)
							,numFaces(0)
							,faceIndex()
							,vertsIndex()
							,verts()
							,normals()
							,st()
							{}

int ObjParser::fixV(int index)const { return(index > 0 ? index - 1 : (index == 0 ? 0 : (int)verts.size() + index)); }
int ObjParser::fixVt(int index)const { return(index > 0 ? index - 1 : (index == 0 ? 0 : (int)st.size() + index)); }
int ObjParser::fixVn(int index)const { return(index > 0 ? index - 1 : (index == 0 ? 0 : (int)normals.size() + index)); }

ObjParser::~ObjParser(){}
Mesh* loadFile( const std::string& filename, const SquareMatrix4f & model ){
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) return nullptr;
    Mesh* mesh = nullptr;
    CGParser* parser = nullptr;
    std::string expantion = filename.substr(pos + 1, 3);
    if (expantion   == "obj"){
			parser = new ObjParser(model);
    }else if(expantion == "geo") {
			parser = new GeoParser(model);
    }
    mesh = parser->readFile(filename.c_str());
    delete parser;
    return mesh;
}
/*! Parse differently formated triplets like: n0, n0/n1/n2, n0//n2, n0/n1.          */
/*! All indices are converted to C-style (from 0). Missing entries are assigned -1. */
VertexProperties ObjParser::getInt3(const char*& token)const
{
	VertexProperties v(-1);
    v.vert.x = fixV(atoi(token));
    token += strcspn(token, "/ \t\r");
    if (token[0] != '/') return(v);
    token++;

    // it is i//n
    if (token[0] == '/') {
        token++;
        v.vert.z = fixVn(atoi(token));
        token += strcspn(token, " \t\r");
        return(v);
    }

    // it is i/t/n or i/t
    v.vert.y = fixVt(atoi(token));
    token += strcspn(token, "/ \t\r");
    if (token[0] != '/') return(v);
    token++;

    // it is i/t/n
    v.vert.z = fixVn(atoi(token));
    token += strcspn(token, " \t\r");
    return(v);
}


/*! Parse separator. */
static inline const char* parseSep(const char*& token) {
    size_t sep = strspn(token, " \t");
    if (!sep) throw std::runtime_error("separator expected");
    return token+=sep;
}
/*! Parse optional separator. */
static inline const char* parseSepOpt(const char*& token) {
    return token+=strspn(token, " \t");
}

/*! Read float from a string. */
static inline float getFloat(const char*& token) {
    token += strspn(token, " \t");
    float n = (float)atof(token);
    token += strcspn(token, " \t\r");
    return n;
}

/*! Read Vec2f from a string. */
static inline Vec2f getVec2f(const char*& token) {
    float x = getFloat(token);
    float y = getFloat(token);
    return Vec2f(x,y);
}
/*! Read Vec3f from a string. */
static inline Vec3f getVec3f(const char*& token) {
    float x = getFloat(token);
    float y = getFloat(token);
    float z = getFloat(token);
    return Vec3f(x, y, z);
}
/*! Determine if character is a separator. */
static inline bool isSep(const char c) {
    return (c == ' ') || (c == '\t');
}

Mesh* ObjParser::readFile(const char *filename){
	std::ifstream ifs;
	std::vector<Vec3f> _normals;//vn массив нормалей к вертексам
	std::vector<Vec2f> _st;//vt массив текстурных координат
	try{
		ifs.open(filename);
		if (ifs.fail()) std::runtime_error("can't open file " + std::string(filename));
		//char line[MAX_LINE_LENGTH];
		std::string line;

		while( !ifs.eof()){

			std::getline(ifs, line);
			const char* token = line.c_str() + std::strspn(line.c_str(), " \t");
			if (token[0] == 0) continue;
			if (token[0] == '#') continue; // ignore comments
            if (token[0] == 'v' && isSep( token[1] ) ) {
            	verts.push_back(getVec3f(token += 2));
            	continue;
            }
            if (!std::strncmp(token, "vn",  2) && isSep(token[2]) )
            {
            	_normals.push_back( getVec3f(token += 3) );
            	continue;
            }
            if (!std::strncmp(token, "vt",  2) && isSep(token[2])) {
            	_st.push_back(getVec2f(token += 3));
            	continue;
            }
            if (token[0] == 'f' && isSep(token[1])) {
                parseSep(token += 1);
                int faceCounter = 0;
                while (token[0]) {
                    face.push_back(getInt3(token));
                	++faceCounter;
                    parseSepOpt(token);
                }
                faceIndex.push_back(faceCounter);

                continue;
            }
            if (!strncmp(token, "usemtl", 6) ) continue;
            if (!strncmp(token, "mtllib", 6) ) continue;
		}
		//go through face, swap texture and normal coordinates
		vertsIndex.reserve( faceIndex.size() );
		normals.reserve( faceIndex.size() );
		st.reserve( faceIndex.size() );
		for(auto It:face){
			vertsIndex.push_back(It.vert.x);
			if(It.vert.y == -1){
				st.push_back( Vec2f(0)  );
			}else{
				st.push_back( _st[It.vert.y]  );
			}
			if(It.vert.z == -1){
				normals.push_back( Vec3f(0)  );
			}else{
				normals.push_back(_normals[It.vert.z] );
			}
		}

		return new Mesh(numFaces, faceIndex, vertsIndex, verts, normals, st, model);
	}
	catch (const std::exception &e) {
		std::cerr << e.what() << std::endl;
		return nullptr;
	}

}
