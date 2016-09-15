TARGET=rTTracer0.9_alpha_demo
INCLUDE_DIR=include/
SRC_DIR=source/
BUILD_DIR=build/
COMPILER=nvcc
COMPILER_FLAGS=-std=c++11 -O3
NVCC_COMMON_FLAGS = --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 
NVCC_COMPILER_FLAGS =-Xcompiler \-fopenmp -D__ASYNCH -D__BOUNDING -D__OMP 
NVCC_LINKER_FLAGS=-Xlinker -lgomp
LIBRARIES=-lGL -lGLEW -lGLU -lglut
all: ${BUILD_DIR}${TARGET}
COMPILE_LIB=${COMPILER} -I${INCLUDE_DIR} ${COMPILER_FLAGS} ${NVCC_COMPILER_FLAGS} ${NVCC_COMMON_FLAGS} -c ${SRC_DIR}
LINK=${COMPILER} ${NVCC_LINKER_FLAGS} ${BUILD_DIR}*.o ${NVCC_COMMON_FLAGS} ${LIBRARIES} -o ${BUILD_DIR}${TARGET}
acceleration.o: ${INCLUDE_DIR}acceleration.cuh ${SRC_DIR}acceleration.cu
	${COMPILE_LIB}acceleration.cu -o ${BUILD_DIR}acceleration.o
#algorithms.o: ${INCLUDE_DIR}algorithms.cuh ${SRC_DIR}algorithms.cu
#	${COMPILE_LIB}algorithms.cu -o ${BUILD_DIR}algorithms.o
bitmap.o: ${INCLUDE_DIR}bitmap.cuh ${SRC_DIR}bitmap.cu
	${COMPILE_LIB}bitmap.cu -o ${BUILD_DIR}bitmap.o
describers.o: ${INCLUDE_DIR}describers.cuh ${SRC_DIR}describers.cu
	${COMPILE_LIB}describers.cu -o ${BUILD_DIR}describers.o
geom.o: ${INCLUDE_DIR}geom.cuh ${SRC_DIR}geom.cu
	${COMPILE_LIB}geom.cu -o ${BUILD_DIR}geom.o
helper.o: ${INCLUDE_DIR}helper.cuh ${SRC_DIR}helper.cu
	${COMPILE_LIB}helper.cu -o ${BUILD_DIR}helper.o
light.o: ${INCLUDE_DIR}light.cuh ${SRC_DIR}light.cu
	${COMPILE_LIB}light.cu -o ${BUILD_DIR}light.o
main.o: ${SRC_DIR}main.cu
	${COMPILE_LIB}main.cu -o ${BUILD_DIR}main.o
objectBox.o: ${INCLUDE_DIR}objectBox.cuh ${SRC_DIR}objectBox.cu
	${COMPILE_LIB}objectBox.cu -o ${BUILD_DIR}objectBox.o
parser.o: ${INCLUDE_DIR}parser.cuh ${SRC_DIR}parser.cu
	${COMPILE_LIB}parser.cu -o ${BUILD_DIR}parser.o
rendering.o: ${INCLUDE_DIR}rendering.cuh ${SRC_DIR}rendering.cu
	${COMPILE_LIB}rendering.cu -o ${BUILD_DIR}rendering.o
transformations.o: ${INCLUDE_DIR}transformations.cuh ${SRC_DIR}transformations.cu
	${COMPILE_LIB}transformations.cu -o ${BUILD_DIR}transformations.o
${BUILD_DIR}${TARGET}: acceleration.o bitmap.o describers.o geom.o helper.o light.o main.o objectBox.o parser.o rendering.o transformations.o
	${LINK}
clean:
	rm -rf ${BUILD_DIR}*.o
