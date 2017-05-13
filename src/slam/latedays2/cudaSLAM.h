#ifndef __CUDA_SLAM_H__
#define __CUDA_SLAM_H__

#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "noise.h"

#include <iostream>

#include <Eigen/Dense>
using namespace Eigen;

struct Particle {
    Matrix4f transform;
};

class CudaSLAM {

private:
    
    float* cudaDeviceLidarArray;
    Particle* cudaDeviceParticles;

public:
    float* myMatrix;
    Particle* particles;
    int numParticles;

    int lidarMaxArraySize;



    float x1; //latest without error
    float y1; //latest without error
    float z1; //latest without error

    

    CudaSLAM(int particleFilterSize, int maxLidar);
    virtual ~CudaSLAM();

    void allocParticles();

    //copy back to CPU
    void retrieveParticles();

    void initParticles();

    void setup();

    
    void printDeviceInfo();

    void doSomethingAwesome();
    
    void doFusedParticleUpdate(float vx, float vy, float vz, float dt,
                              float x0, float y0, float z0, 
                              float roll, float pitch, float yaw);

    //void runMapCompares(float* lidar_new, int new_length, float* lidar_old, int old_length);

    void pushLidar(float* lidar_array, int array_length);
    void transformLidar(int array_length, int particle_id);
    void retrieveLidar(float* lidar_array, int array_length);

    void transformLidarSequential(float* lidar_array, int array_length, int particle_id);
};




#endif