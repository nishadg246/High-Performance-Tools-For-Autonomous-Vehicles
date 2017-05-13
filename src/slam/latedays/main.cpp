#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <math.h>

#include <Eigen/Dense>

#include "cycleTimer.h"
#include "cudaSLAM.h"
#include <iostream>

#define NUM_PARTICLES 1000*10

using Eigen::MatrixXd;

int main(int argc, char** argv)
{

    // MatrixXd m;
    // m = MatrixXd::Zero(4,4);
    // m(0,0) = 3;
    // m(1,0) = 2.5;
    // m(0,1) = -1;
    // m(1,1) = m(1,0) + m(0,1);
    // std::cout << m << std::endl;
    int repetitions = 10;

    printf("main() started\n");

   





    CudaSLAM* worker;
    worker = new CudaSLAM(NUM_PARTICLES);
    worker->allocParticles();
    worker->setup();
    worker->initParticles();

    float dt = 1.0;
    float vx = 2.0, vy = 1.0, vz = -1.0;
    float x0 = 0.0, y0 = 0.0, z0 = 0.0;
    float roll = 3.14/2, pitch = 0.0, yaw = -3.14/2;

    double startClearTime = CycleTimer::currentSeconds();

    for(int i = 0; i < repetitions; i++){
        worker->doFusedParticleUpdate(vx, vy, vz, dt,
                                    x0, y0, z0, 
                                    roll, pitch, yaw);
    }

    double endClearTime = CycleTimer::currentSeconds();

    // worker->setParticlePositions(5,6,7);
    // worker->offsetParticlePositions(15,14,13);

    // worker->setParticleAngles(5,6,7);
    // worker->offsetParticleAngles(15,14,13);
    
    worker->getParticles();


    
    printf("main() ended at %f\n",(endClearTime - startClearTime));
    printf(" ..but repeated 'doFused' %d times\n",repetitions);
    printf(" ..over %d points\n",NUM_PARTICLES);
    printf(" did each fuse in average time of %f\n", ((endClearTime - startClearTime))/repetitions);
    return 0;
}

