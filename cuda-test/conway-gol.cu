#include <algorithm>
#include <assert.h>

__global__ void simpleLifeKernel(
    const unsigned char* lifeData,
    unsigned int worldWidth,
    unsigned int worldHeight,
    unsigned char* resultLifeData
  ){
    unsigned int worldSize = worldWidth * worldHeight;
 
    for (unsigned int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      cellId < worldSize;
      cellId += blockDim.x * gridDim.x
    ){
      unsigned int x = cellId % worldWidth;
      unsigned int yAbs = cellId - x;
      unsigned int xLeft = (x + worldWidth - 1) % worldWidth;
      unsigned int xRight = (x + 1) % worldWidth;
      unsigned int yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
      unsigned int yAbsDown = (yAbs + worldWidth) % worldSize;

      unsigned int aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
        + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
        + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

      resultLifeData[x + yAbs] =  aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
  }
}

void runSimpleLifeKernel(
    unsigned char*& d_lifeData,
    unsigned char*& d_lifeDataBuffer,
    size_t worldWidth,
    size_t worldHeight,
    size_t iterationsCount,
    ushort threadsCount
  ){
    assert((worldWidth * worldHeight) % threadsCount == 0);
    size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
    ushort blocksCount = (ushort )std::min((size_t) 32768, reqBlocksCount);
    for (size_t i = 0; i < iterationsCount; ++i) {
      simpleLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, worldWidth,
      worldHeight, d_lifeDataBuffer);
      std::swap(d_lifeData, d_lifeDataBuffer);
    }
}

int main(){
  //runSimpleLifeKernel()
  return 0;
}