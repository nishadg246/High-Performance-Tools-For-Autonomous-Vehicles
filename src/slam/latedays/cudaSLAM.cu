#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "noise.h"

#include "cudaSLAM.h"

////////////////////////////////  DIAGNOSTICS  ///////////////////////////////////////////
    void 
    CudaSLAM::printDeviceInfo(){
        int deviceCount = 0;
        std::string name;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        printf("---------------------------------------------------------\n");
        printf("Initializing CUDA for CudaRenderer\n");
        printf("Found %d CUDA devices\n", deviceCount);

        for (int i=0; i<deviceCount; i++) {
            cudaDeviceProp deviceProps;
            cudaGetDeviceProperties(&deviceProps, i);
            name = deviceProps.name;

            printf("Device %d: %s\n", i, deviceProps.name);
            printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
            printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
            printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        }
        printf("---------------------------------------------------------\n");
    }

//////////////////////////////////////////////////////////////////////////////////////////


struct GlobalConstants {
    int numParticles;
    Particle* particles;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;




////////////////////////////////  NOISE  ///////////////////////////////////////////

    // read-only lookup tables used to quickly compute noise (needed by
    // advanceAnimation for the snowflake scene)
    __constant__ int    cuConstShuffledTableX[256];
    __constant__ int    cuConstShuffledTableY[256];
    __constant__ int    cuConstShuffledTableZ[256];
    __constant__ int    cuConstShuffledTableRoll[256];
    __constant__ int    cuConstShuffledTablePitch[256];
    __constant__ int    cuConstShuffledTableYaw[256];
    __constant__ float  cuConstNoise1DUniformTable[256];
    __constant__ float  cuConstNoise1DNormalTable[256];

    #include "noiseCuda.cu_inl"

    void setupNoise(){
        // also need to copy over the noise lookup tables, so we can
        // implement noise on the GPU
        int* shuffle1;
        int* shuffle2;
        int* shuffle3;
        int* shuffle4;
        int* shuffle5;
        int* shuffle6;
        float* noise1;
        float* noise2;
        getShufflePositionTables(&shuffle1, &shuffle2, &shuffle3); 
        getShuffleAngleTables(&shuffle4, &shuffle5, &shuffle6);
        getNoiseTables(&noise1, &noise2);
        cudaMemcpyToSymbol(cuConstShuffledTableX,      shuffle1, sizeof(int)   * 256);
        cudaMemcpyToSymbol(cuConstShuffledTableY,      shuffle2, sizeof(int)   * 256);
        cudaMemcpyToSymbol(cuConstShuffledTableZ,      shuffle3, sizeof(int)   * 256);
        cudaMemcpyToSymbol(cuConstShuffledTableRoll,   shuffle4, sizeof(int)   * 256);
        cudaMemcpyToSymbol(cuConstShuffledTablePitch,  shuffle5, sizeof(int)   * 256);
        cudaMemcpyToSymbol(cuConstShuffledTableYaw,    shuffle6, sizeof(int)   * 256);
        cudaMemcpyToSymbol(cuConstNoise1DUniformTable, noise1,   sizeof(float) * 256);
        cudaMemcpyToSymbol(cuConstNoise1DNormalTable,  noise2,   sizeof(float) * 256);
    }

//////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////  INITIALIZATION  ////////////////////////////////////////


CudaSLAM::CudaSLAM(int particleFilterSize) {
    numParticles = particleFilterSize;
    particles = NULL;
    cudaDeviceParticles = NULL;
}

CudaSLAM::~CudaSLAM() {

    if (particles) {
        delete [] particles;
    }

    if (cudaDeviceParticles) {
        cudaFree(cudaDeviceParticles);
    }
}

void
CudaSLAM::allocParticles(){
    if (particles)
        delete particles;
    particles = new Particle[numParticles];
}

void 
CudaSLAM::setup(){
    //printDeviceInfo();
    setupNoise();


    cudaMalloc(&cudaDeviceParticles, sizeof(Particle) * numParticles);
    cudaMemcpy(cudaDeviceParticles, particles, sizeof(Particle) * numParticles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.numParticles = numParticles;
    params.particles = cudaDeviceParticles;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
}

//////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////  KERNEL TEMPLATES  ////////////////////////////////////////

    __device__ __inline__ void
    awesomeHelper() {

    }

    __global__ void kernelAwesome() {
        awesomeHelper();

    }

//////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////  PARTICLES OPS ////////////////////////////////////////

// __global__ void
// kernelInitParticles(){
//     uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);
//     if(index < cuConstRendererParams.numParticles){
//         Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);
//         p->x = 0.0;
//         p->y = 0.0;
//         p->z = 0.0;
//         p->roll = 0.0;
//         p->pitch = 0.0;
//         p->yaw = 0.0;    
//     }
// }

__global__ void
kernelInitParticles(){
    uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if(index < cuConstRendererParams.numParticles){
        Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);

        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                p->transform(i,j) = 0;
            }
        }
    }
}

///////////////////////// POSITION SET and OFFSET ////////////////////////////////////////

    // __global__ void
    // kernelSetParticlePositions(float x, float y, float z){
    //     uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);
    //     if(index < cuConstRendererParams.numParticles){
    //         Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);
    //         p->x = x;
    //         p->y = y;
    //         p->z = z;
    //     }
    // }

    // __global__ void
    // kernelOffsetParticlePositions(float d_x, float d_y, float d_z){
    //     uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);
    //     if(index < cuConstRendererParams.numParticles){
    //         Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);
    //         p->x += d_x;
    //         p->y += d_y;
    //         p->z += d_z;   
    //     }
    // }

///////////////////////// ANGLES SET and OFFSET ////////////////////////////////////////

    // __global__ void
    // kernelSetParticleAngles(float roll, float pitch, float yaw){
    //     uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);
    //     if(index < cuConstRendererParams.numParticles){
    //         Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);
    //         p->roll = roll;
    //         p->pitch = pitch;
    //         p->yaw = yaw;    
    //     }
    // }

    // __global__ void
    // kernelOffsetParticleAngles(float d_roll, float d_pitch, float d_yaw){
    //     uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);
    //     if(index < cuConstRendererParams.numParticles){
    //         Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);
    //         p->roll += d_roll;
    //         p->pitch += d_pitch;
    //         p->yaw += d_yaw;     
    //     }
    // }

//////////////////////////// GENERIC PARTICLE OPS ////////////////////////////////////////

    void 
    CudaSLAM::getParticles() {
        cudaMemcpy(particles,
                   cudaDeviceParticles,
                   sizeof(Particle) * numParticles,
                   cudaMemcpyDeviceToHost);

        //for(int i = 0; i < numParticles; i++){
        std::cout << particles[0].transform << std::endl << std::endl;
            //printf("Particle %d: x,y,z = (%5f,%5f,%5f) r,p,y = (%5f,%5f,%5f)\n",i,particles[i].x,particles[i].y,particles[i].z,particles[i].roll,particles[i].pitch,particles[i].yaw);
        //}
    }

    void
    CudaSLAM::initParticles() {
        printf("initParticles\n");

        int blockLength = 256;
        int numTiles = (numParticles)/blockLength;

        dim3 threadsPerBlock(blockLength,1);
        dim3 blocksPerGrid(numTiles, 1, 1);

        kernelInitParticles<<<blocksPerGrid,threadsPerBlock>>>();
        cudaDeviceSynchronize();
    }



__global__ void
kernelAddRotate(float3 a){
    uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if(index < cuConstRendererParams.numParticles){
        Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);

        float roll_error = 0.0*cudaGetRollNoise(index);
        float pitch_error = 0.0*cudaGetPitchNoise(index);
        float yaw_error = 0.0*cudaGetYawNoise(index);

        Matrix3f Rx, Ry, Rz, R;

        float c, s;

        c = cosf(a.x + roll_error);
        s = sinf(a.x + roll_error);

        Rx << 1, 0, 0,
              0, c, -s,
              0, s, c;

        c = cosf(a.y + pitch_error);
        s = sinf(a.y + pitch_error);

        Ry << c, 0, s,
              0, 1, 0,
              -s, 0, c;

        c = cosf(a.z + yaw_error);
        s = sinf(a.z + yaw_error);

        Rz << c, -s, 0,
              s, c, 0,
              0, 0, 1;

        R = Rz*(Ry*Rx);

        //p->transform = R;


        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                p->transform(i,j) = R(i,j);
            }
        }
        p->transform(3,3) = 1.0;
    }
}

__global__ void
kernelRotateVelocities(float3 v, float3 p0, float dt){
    uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if(index < cuConstRendererParams.numParticles){
        Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);


        float dx = (v.x + (0.0*cudaGetXNoise(index))) * dt;
        float dy = (v.y + (0.0*cudaGetYNoise(index))) * dt;
        float dz = (v.z + (0.0*cudaGetZNoise(index))) * dt;
        Vector4f d1(dx,dy,dz,1.0);

        Vector4f d2 = (p->transform) * d1;
        Vector4f p1(p0.x + d2(0), p0.y + d2(1), p0.z + d2(2),1.0);


        for(int i = 0; i < 3; i++){
            p->transform(i,3) = p1(i);
        }
    }
}

void
CudaSLAM::doFusedParticleUpdate(float vx, float vy, float vz, float dt,
                              float x0, float y0, float z0, 
                              float roll, float pitch, float yaw) {

    //printf("startFusedParticles\n");

    int blockLength = 128;
    int numTiles = (numParticles + (blockLength-1))/blockLength;



    dim3 threadsPerBlock(blockLength,1);
    dim3 blocksPerGrid(numTiles, 1, 1);

    float3 p0 = make_float3(x0,y0,z0);
    float3 v = make_float3(vx,vy,vz);
    float3 a = make_float3(roll,pitch,yaw);

    kernelAddRotate<<<blocksPerGrid,threadsPerBlock>>>(a);
    kernelRotateVelocities<<<blocksPerGrid,threadsPerBlock>>>(v, p0, dt);
    cudaDeviceSynchronize();
}

    // void
    // CudaSLAM::setParticlePositions(float x, float y, float z) {
    //     printf("setParticlePositions\n");

    //     int blockLength = 256;
    //     int numTiles = (numParticles)/blockLength;

    //     dim3 threadsPerBlock(blockLength,1);
    //     dim3 blocksPerGrid(numTiles, 1, 1);

    //     kernelSetParticlePositions<<<blocksPerGrid,threadsPerBlock>>>(x,y,z);
    //     cudaDeviceSynchronize();
    // }

    // void
    // CudaSLAM::offsetParticlePositions(float d_x, float d_y, float d_z) {
    //     printf("offsetParticlePositions\n");

    //     int blockLength = 256;
    //     int numTiles = (numParticles)/blockLength;

    //     dim3 threadsPerBlock(blockLength,1);
    //     dim3 blocksPerGrid(numTiles, 1, 1);

    //     kernelOffsetParticlePositions<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_z);
    //     cudaDeviceSynchronize();
    // }

    // void
    // CudaSLAM::setParticleAngles(float roll, float pitch, float yaw) {
    //     printf("setParticleAngles\n");

    //     int blockLength = 256;
    //     int numTiles = (numParticles)/blockLength;

    //     dim3 threadsPerBlock(blockLength,1);
    //     dim3 blocksPerGrid(numTiles, 1, 1);

    //     kernelSetParticleAngles<<<blocksPerGrid,threadsPerBlock>>>(roll,pitch,yaw);
    //     cudaDeviceSynchronize();
    // }

    // void
    // CudaSLAM::offsetParticleAngles(float d_roll, float d_pitch, float d_yaw) {
    //     printf("offsetParticleAngles\n");

    //     int blockLength = 256;
    //     int numTiles = (numParticles)/blockLength;

    //     dim3 threadsPerBlock(blockLength,1);
    //     dim3 blocksPerGrid(numTiles, 1, 1);

    //     kernelOffsetParticleAngles<<<blocksPerGrid,threadsPerBlock>>>(d_roll,d_pitch,d_yaw);
    //     cudaDeviceSynchronize();
    // }

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////// MATRIX PARTICLE OPS ////////////////////////////////////////

// void
// CudaSLAM::offsetParticleAngles(float d_roll, float d_pitch, float d_yaw) {
//     printf("offsetParticleAngles\n");

//     int blockLength = 256;
//     int numTiles = (numParticles)/blockLength;

//     dim3 threadsPerBlock(blockLength,1);
//     dim3 blocksPerGrid(numTiles, 1, 1);

//     kernelOffsetParticleAngles<<<blocksPerGrid,threadsPerBlock>>>(d_roll,d_pitch,d_yaw);
//     cudaDeviceSynchronize();
// }

//////////////////////////////////////////////////////////////////////////////////////////

// void
// CudaSLAM::doSomethingAwesome() {
//     setupNoise();
// }

//////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////

// __global__ void staticReverse(int *d, int n)
// {
//   __shared__ int s[64];
//   int t = threadIdx.x;
//   int tr = n-t-1;
//   s[t] = d[t];
//   __syncthreads();
//   d[t] = s[tr];
// }

// void reverseExample(){
    
//     const int n = 64;
//     int a[n], d[n];

//     for (int i = 0; i < n; i++) {
//         a[i] = i;
//         d[i] = 0;
//     }

//     int *d_d;
//     cudaMalloc(&d_d, n * sizeof(int)); 

//     cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
//     staticReverse<<<1,n>>>(d_d, n);
//     cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);

//     for (int i = 0; i < n; i++)
//         printf("d[%d] = %d\n",i,d[i]);
        
// }


// __global__ void randomArray(int *d)
// {
//     int t = threadIdx.x;
//     float random = cudaFloat1DNormalNoise(t);
//     d[t] = static_cast<int>(1000.0*random);

//     //__shared__ int s[64];
//     //int t = threadIdx.x;
//     //int tr = n-t-1;
//     //s[t] = d[t];
//     //__syncthreads();
//     //d[t] = s[tr];
// }

// void testNoise(){
//         setupNoise();

//     const int n = 64;
//     int d[n];

//     for (int i = 0; i < n; i++) {
//         d[i] = 0;
//     }

//     int *d_d;
//     cudaMalloc(&d_d, n * sizeof(int)); 
//     randomArray<<<1,n>>>(d_d);
//     cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);

//     for (int i = 0; i < n; i++)
//         printf("%d,\n",d[i]);
// }