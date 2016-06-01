nvcc -g -G -std=c++11 -arch=sm_50 -rdc=true -lcudadevrt $1.cu -o $1.exe
