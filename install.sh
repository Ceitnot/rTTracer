#!bash
nvcc_version=`which nvcc`
cc_version=`which nvcc`
if [[ "$nvcc_version" == "" ]]
then
#elif [[  ]]
	echo "There is no usable nvcc cuda compiler on the system. Please, install one and try again."
else
	mkdir build
	make
	make clean
fi
