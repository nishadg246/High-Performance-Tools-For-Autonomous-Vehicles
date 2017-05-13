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
    int numParticles;
    Particle* particles;
    Particle* cudaDeviceParticles;

public:

    CudaSLAM(int particleFilterSize);
    virtual ~CudaSLAM();

    void allocParticles();
    void getParticles();
    // void setParticlePositions(float x, float y, float z);
    // void offsetParticlePositions(float d_x, float d_y, float d_z);
    // void setParticleAngles(float roll, float pitch, float yaw);
    // void offsetParticleAngles(float d_roll, float d_pitch, float d_yaw);
    void initParticles();

    void setup();

    
    void printDeviceInfo();

    void doSomethingAwesome();
    
    void doFusedParticleUpdate(float vx, float vy, float vz, float dt,
                              float x0, float y0, float z0, 
                              float roll, float pitch, float yaw);
};




#endif