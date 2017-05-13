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
    float* lidarArray;
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


CudaSLAM::CudaSLAM(int particleFilterSize, int maxLidar) {
    numParticles = particleFilterSize;
    lidarMaxArraySize = maxLidar;

    particles = NULL;
    // lidarArray = NULL;

    myMatrix = NULL;

    cudaDeviceParticles = NULL;
    cudaDeviceLidarArray = NULL;

    x1 = 0.0;
    y1 = 0.0;
    z1 = 0.0;
}

CudaSLAM::~CudaSLAM() {

    if (particles) {
        delete [] particles;
    }

    if(myMatrix) {
        delete [] myMatrix;
    }

    if (cudaDeviceParticles) {
        cudaFree(cudaDeviceParticles);
    }

    if (cudaDeviceLidarArray) {
        cudaFree(cudaDeviceLidarArray);
    }
}

void
CudaSLAM::allocParticles(){
    if (particles)
        delete particles;
    particles = new Particle[numParticles];

    // if (lidarArray)
    //     delete lidarArray;
    // lidarArray = new float[lidarArraySize];

    if (myMatrix)
        delete myMatrix;
    myMatrix = new float[4*4];

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            if(i == j){
                myMatrix[(j*4)+i] = 1.0;
            } else{
                myMatrix[(j*4)+i] = 0.0;
            }
        }
    }


}

void 
CudaSLAM::setup(){
    //printDeviceInfo();
    setupNoise();


    cudaMalloc(&cudaDeviceParticles, sizeof(Particle) * numParticles);
    cudaMemcpy(cudaDeviceParticles, particles, sizeof(Particle) * numParticles, cudaMemcpyHostToDevice);

    cudaMalloc(&cudaDeviceLidarArray, 4 * sizeof(float) * lidarMaxArraySize);

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
    params.lidarArray = cudaDeviceLidarArray;

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
                if(i == j){
                    p->transform(j,i) = 1.0;
                } else{
                    p->transform(j,i) = 0.0;
                }
            }
        }
    }
}

//////////////////////////// GENERIC PARTICLE OPS ////////////////////////////////////////

    void 
    CudaSLAM::retrieveParticles() {
        cudaMemcpy(particles,
                   cudaDeviceParticles,
                   sizeof(Particle) * numParticles,
                   cudaMemcpyDeviceToHost);

       
        x1 = particles[0].transform(0,3);
        y1 = particles[0].transform(1,3);
        z1 = particles[0].transform(2,3);

        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                myMatrix[(j*4)+i] = particles[0].transform(j,i);          
            }
        }
        
        
        //for(int i = 0; i < numParticles; i++){
        //std::cout << particles[0].transform << std::endl << std::endl;
            //printf("Particle %d: x,y,z = (%5f,%5f,%5f) r,p,y = (%5f,%5f,%5f)\n",i,particles[i].x,particles[i].y,particles[i].z,particles[i].roll,particles[i].pitch,particles[i].yaw);
        //}
    }

    void
    CudaSLAM::initParticles() {

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

    float roll_error = 0.0;
    float pitch_error = 0.0;
    float yaw_error = 0.0;

    if(index < cuConstRendererParams.numParticles){
        Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);

        if(index != 0){
            float roll_error = 1.0*cudaGetRollNoise(index);
            float pitch_error = 1.0*cudaGetPitchNoise(index);
            float yaw_error = 1.0*cudaGetYawNoise(index);            
        }

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
        //reset translational component

        p->transform(0,3) = 0.0;
        p->transform(1,3) = 0.0;
        p->transform(2,3) = 0.0;
        p->transform(3,3) = 1.0;
    }
}

__global__ void
kernelRotateVelocities(float3 v, float3 p0, float dt){
    uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);

    float vx_error = 0.0;
    float vy_error = 0.0;
    float vz_error = 0.0;

    if(index < cuConstRendererParams.numParticles){
        Particle* p = (Particle*)(&cuConstRendererParams.particles[index]);

        if(index != 0){
            vx_error = (1.0*cudaGetXNoise(index));
            vy_error = (1.0*cudaGetYNoise(index));
            vz_error = (1.0*cudaGetZNoise(index));            
        }

        float dx = (v.x + vx_error) * dt;
        float dy = (v.y + vy_error) * dt;
        float dz = (v.z + vz_error) * dt;

        Vector4f d1(dx,dy,dz,1.0);

        Vector4f d2 = (p->transform) * d1;
        Vector4f p1(p0.x + d2(0), p0.y + d2(1), p0.z + d2(2),1.0);
        //Vector4f p1(d2(0), d2(1), d2(2),1.0);

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

    if(dt >= 0.0){
        kernelAddRotate<<<blocksPerGrid,threadsPerBlock>>>(a);
        kernelRotateVelocities<<<blocksPerGrid,threadsPerBlock>>>(v, p0, dt);
        cudaDeviceSynchronize();        
    } else{
        kernelAddRotate<<<blocksPerGrid,threadsPerBlock>>>(a);
        cudaDeviceSynchronize();     
    }


}





void
CudaSLAM::pushLidar(float* lidar_array, int array_length){
    cudaMemcpy(cudaDeviceLidarArray, lidar_array, 4 * sizeof(float) * array_length, cudaMemcpyHostToDevice);
}

__global__ void
kernelTransformLidar(int array_length, int particle_id){
    uint index = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if(index < array_length){

        float* x = (float*)(&cuConstRendererParams.lidarArray[(4*index)]);
        float* y = (float*)(&cuConstRendererParams.lidarArray[(4*index)+1]);
        float* z = (float*)(&cuConstRendererParams.lidarArray[(4*index)+2]);

        Vector4f a(*x,*y,*z,1.0);
        Particle* p = (Particle*)(&cuConstRendererParams.particles[particle_id]);

        Vector4f b = p->transform * a;

        *x = b(0);
        *y = b(1);
        *z = b(2);
    }
}

void
CudaSLAM::transformLidar(int array_length, int particle_id){
    int blockLength = 1024;
    int numTiles = (array_length + (blockLength - 1))/blockLength;

    dim3 threadsPerBlock(blockLength,1);
    dim3 blocksPerGrid(numTiles, 1, 1);   

    kernelTransformLidar<<<blocksPerGrid,threadsPerBlock>>>(array_length,particle_id);
    cudaDeviceSynchronize();

}

void
CudaSLAM::transformLidarSequential(float* lidar_array, int array_length, int particle_id){

    for(int i = 0; i < array_length; i++){
        float x = lidar_array[(4*i)];
        float y = lidar_array[(4*i)+1];
        float z = lidar_array[(4*i)+2];

        Vector4f a(x,y,z,1.0);

        Vector4f b = particles[particle_id].transform * a;
        lidar_array[(4*i)]   = b(0);
        lidar_array[(4*i)+1] = b(1);
        lidar_array[(4*i)+2] = b(2);    
    }
}

void
CudaSLAM::retrieveLidar(float* lidar_array, int array_length){
    cudaMemcpy(lidar_array,
               cudaDeviceLidarArray,
               4 * sizeof(float) * array_length,
               cudaMemcpyDeviceToHost);    


}



