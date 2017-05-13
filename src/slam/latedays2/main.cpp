#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <math.h>

#include <Eigen/Dense>

#include "cycleTimer.h"
#include "cudaSLAM.h"
#include <iostream>
#include <string>


#include <iostream>
#include <fstream>


#define MAX_FRAMES (2)
//#define MAX_FRAMES (5)

#define NUM_PARTICLES (100)

//#define MAX_LIDAR_POINTS (100*1000)
#define MAX_LIDAR_POINTS (100)

using Eigen::MatrixXd;

using namespace std;



void zeroOutLidarArray(float* lidar_array, int length){
    if(lidar_array != NULL){
        //zero out lidar data
        for(int i = 0; i < length; i++){
            lidar_array[(4*i)]     = 0.0;
            lidar_array[(4*i) + 1] = 0.0;
            lidar_array[(4*i) + 2] = 0.0;
            lidar_array[(4*i) + 3] = 0.0;
        }
    }  
}

void printOutLidarArray(float* lidar_array, int length){
    if(lidar_array != NULL){
        //zero out lidar data
        for(int i = 0; i < length; i++){
            float x = lidar_array[(4*i)];
            float y = lidar_array[(4*i) + 1];
            float z = lidar_array[(4*i) + 2];

            printf("%d: (%f,%f,%f)\n", i, x, y, z);
        }
    }  
}



float compareLIDAR(Matrix4f old_transform, Matrix4f new_transform,
                   float* lidar_old, int lidar_old_length,
                   float* lidar_new, int lidar_new_length){

    float gridMetersWidth = 200.0;
    float gridMetersHeight = 200.0;

    const int gridBinsX = 10;
    const int gridBinsY = 10;

    float metersPerBinX = gridMetersWidth/gridBinsX;
    float metersPerBinY = gridMetersHeight/gridBinsY;

    int* grid_old = (int*)malloc(gridBinsX * gridBinsY * sizeof(int));
    int* grid_new = (int*)malloc(gridBinsX * gridBinsY * sizeof(int)); 

    
    float dx, dy;
    int bin_x, bin_y;       

    for(int i = 0; i < (gridBinsX * gridBinsY); i++){
        grid_old[i] = 0;
        grid_new[i] = 0;     
    }

    for(int k = 0; k < lidar_old_length; k++){
        Vector4f v0(lidar_old[(4*k)],lidar_old[(4*k)+1],lidar_old[(4*k)+2],1.0);
        Vector4f v1 = old_transform * v0;

        dx = (v1(0) - old_transform(0,3));
        dy = (v1(1) - old_transform(1,3));

        bin_x = (int)((dx/(metersPerBinX)) + gridBinsX/2);
        bin_y = (int)((dy/(metersPerBinY)) + gridBinsY/2);

        if((bin_x >= 0)&&(bin_x < gridBinsX)){
            if((bin_y >= 0)&&(bin_y < gridBinsY)){
                grid_old[(bin_y * gridBinsX) + bin_x] += 1;
            }                    
        }
    }

    for(int k = 0; k < lidar_new_length; k++){
        Vector4f v0(lidar_new[(4*k)],lidar_new[(4*k)+1],lidar_new[(4*k)+2],1.0);
        Vector4f v1 = new_transform * v0;

        dx = (v1(0)- old_transform(0,3));
        dy = (v1(1) - old_transform(1,3));

        bin_x = (int)((dx/(metersPerBinX)) + gridBinsX/2);
        bin_y = (int)((dy/(metersPerBinY)) + gridBinsY/2);

        if((bin_x >= 0)&&(bin_x < gridBinsX)){
            if((bin_y >= 0)&&(bin_y < gridBinsY)){
                grid_new[(bin_y * gridBinsX) + bin_x] += 1;
            }                    
        }
    }

    printf("old: \n");
    for(int j = 0; j < gridBinsY; j++){
        for(int i = 0; i < gridBinsX; i++){
            printf("%d,",grid_old[(j * gridBinsX) + i]);
        }        
    }

    printf("new: \n");
    for(int j = 0; j < gridBinsY; j++){
        for(int i = 0; i < gridBinsX; i++){
            printf("%d,",grid_new[(j * gridBinsX) + i]);
        }        
    }

    delete[] grid_old;
    delete[] grid_new;

    return 0.0;
}


int main(int argc, char** argv)
{

    ofstream myfile;
    myfile.open ("log.txt");

    printf("main() started\n");
   
    CudaSLAM* worker;

    worker = new CudaSLAM(NUM_PARTICLES, MAX_LIDAR_POINTS);
    printf(".1\n");

    worker->allocParticles();
    printf(".2\n");
    worker->setup();
    printf(".3\n");
    worker->initParticles();
    printf(".4\n");


    float* scores = (float*)malloc(sizeof(float) * (worker->numParticles));

    printf(".5\n");
    float* lidar_old = NULL;
    float* lidar_new = NULL;
    float* lidar_temp = NULL;
    int lidar_temp_length = 0;
    int lidar_new_length = 0;
    int lidar_old_length = 0;

    lidar_old = (float*)malloc(4 * sizeof(float) * (MAX_LIDAR_POINTS));
    lidar_new = (float*)malloc(4 * sizeof(float) * (MAX_LIDAR_POINTS));
    lidar_new_length = MAX_LIDAR_POINTS;
    lidar_old_length = MAX_LIDAR_POINTS;

    zeroOutLidarArray(lidar_old, MAX_LIDAR_POINTS);
    zeroOutLidarArray(lidar_new, MAX_LIDAR_POINTS);

    printf("allocated lidar arrays\n");

    worker->doFusedParticleUpdate(5, 15, 25, 1,
                                  100.0,10,1, 
                                  3.141592,0,0);

    worker->retrieveParticles();

    printf("retrieved particles\n");

    
    int particle_id = 0;

    //worker->pushLidar(lidar_new, lidar_new_length);
    //printf("pushing %d\n",lidar_new_length);

    //worker->retrieveLidar(lidar_new, lidar_new_length);

    double startClearTime = CycleTimer::currentSeconds();
    for(int i = 0; i < 10; i++){
        worker->transformLidarSequential(lidar_new, lidar_new_length, particle_id);
    }
    double endClearTime = CycleTimer::currentSeconds();

    
    printf("retrieved %d\n",lidar_new_length);


    // Matrix4f old_transform;
    // for(int i = 0; i < 4; i++){
    //     for(int j = 0; j < 4; j++){
    //         if(i == j){
    //             old_transform(j,i) = 1.0;
    //         } else{
    //             old_transform(j,i) = 0.0;
    //         }
    //     }
    // }


    // //FILE IO
    //     FILE* stream_lidar;
    //     FILE* stream_IMU;

    //     int num_lidar_points_read = 0;
    //     int num_IMU_read = 0;

    //     char* buffer = new char[256];
    //     string file_lidar = "";
    //     string file_lidar_path = "../kitti_00/velodyne_points/data/";
    //     string file_lidar_extension = ".bin";

    //     string file_IMU = "";
    //     string file_IMU_path = "../kitti_00/oxts/data/";
    //     string file_IMU_extension = ".txt";

    // float x0 = 0.0, y0 = 0.0, z0 = 0.0;

    // float roll_0 = 0.0, pitch_0 = 0.0, yaw_0 = 0.0;
    // float roll = 0.0, pitch = 0.0, yaw = 0.0;
    // float vx = 0.0, vy = 0.0, vz = 0.0;
    
    // float dt = 0.1;

    // printf("start frames\n");

    // for(int t = 0; t < MAX_FRAMES; t++){

    //     lidar_temp = lidar_old;
    //     lidar_temp_length = lidar_old_length;

    //     lidar_old = lidar_new;
    //     lidar_old_length = lidar_new_length;

    //     lidar_new = lidar_temp;
    //     lidar_new_length = lidar_temp_length;



    //     zeroOutLidarArray(lidar_new, lidar_new_length);

    //     sprintf(buffer, "%s%010d%s",file_lidar_path.c_str(), t, file_lidar_extension.c_str());
    //     file_lidar = buffer;

    //     sprintf(buffer, "%s%010d%s",file_IMU_path.c_str(), t, file_IMU_extension.c_str());
    //     file_IMU = buffer;

    //     //LIDAR at step t
            
    //         num_lidar_points_read = 0;
        
    //         stream_lidar = fopen (file_lidar.c_str(), "rb");
    //         if (stream_lidar == NULL) {
    //            printf("Failed to open: %s\n", file_lidar.c_str());
    //            return -1;
    //         }
    //         num_lidar_points_read = fread(lidar_new, sizeof(float), (MAX_LIDAR_POINTS), stream_lidar)/4;
    //         lidar_new_length = num_lidar_points_read;
            
    //         for(int i = 0; i < MAX_LIDAR_POINTS; i++){
    //             lidar_new[(4*i)]   = 1;
    //             lidar_new[(4*i)+1] = 2;
    //             lidar_new[(4*i)+2] = 3;
    //         }


    //     //IMU at step t
    //         num_IMU_read = 0;

    //         stream_IMU = fopen(file_IMU.c_str(), "rb");
    //         if (stream_IMU == NULL) {
    //            printf("Failed to open: %s\n", file_IMU.c_str());
    //            return -2;
    //         }

    //         num_IMU_read = fscanf(stream_IMU,"%*f %*f %*f %f %f %f %*f %*f %f %f %f",&roll, &pitch, &yaw, &vx, &vy, &vz);
    //         if (num_IMU_read < 6){
    //            printf("Failed to open 6 sensor readings from: %s\n", file_IMU.c_str());
    //            return -3; 
    //         }
    
        

    //     // if(t == 0){
    //     //     worker->doFusedParticleUpdate(vx, vy, vz, -0.0,
    //     //                                   x0, y0, z0, 
    //     //                                   roll, pitch, yaw);
    //     // } else{
    //     //     worker->doFusedParticleUpdate(vx, vy, vz, dt,
    //     //                                   x0, y0, z0, 
    //     //                                   roll, pitch, yaw);
    //     // }

    //     worker->doFusedParticleUpdate(1, 0, 0, dt,
    //                                   0,0,0, 
    //                                   0,0,0);

    //     worker->retrieveParticles();

    //     if(t == 1){
    //         int particle_id = 0;

    //         worker->pushLidar(lidar_new, lidar_new_length);

    //         double startClearTime = CycleTimer::currentSeconds();
    //         worker->transformLidar(lidar_new_length, particle_id);
    //         double endClearTime = CycleTimer::currentSeconds();

    //         worker->retrieveLidar(lidar_new, lidar_new_length);
    //     }

        // if(t == 1){
        //     for(int p = 0; p < worker->numParticles; p++){

        //         // scores[p] = compareLIDAR(old_transform, worker->particles[p].transform,
        //         //     lidar_old,lidar_old_length,lidar_new,lidar_new_length);
        //         scores[p] = 0.0;

        //     }
        //     for(int p = 0; p < worker->numParticles; p++){
        //         printf("%d: %f\n",p,scores[p]);
        //     }
        // }

        // x0 = worker->x1;
        // y0 = worker->y1;
        // z0 = worker->z1;

        
        // for(int j = 0; j < 4; j++){
        //     for(int i = 0; i < 4; i++){
        //         myfile << (worker->myMatrix[(j*4)+i]) << ",";
        //     }
        //     myfile << "\n";
        // }

        //printf("%f, %f, %f\n",x0,y0,z0);

        
//    }
    free((void*)lidar_old);
    free((void*)lidar_new);
    //delete[] buffer;

    //myfile.close();
    printf("timer: %f\n",(endClearTime - startClearTime)/10.0);

    
    // printf("main() ended at %f\n",(endClearTime - startClearTime));
    // printf(" ..but repeated 'doFused' %d times\n",repetitions);
    // printf(" ..over %d points\n",NUM_PARTICLES);
    // printf(" did each fuse in average time of %f\n", ((endClearTime - startClearTime))/repetitions);
    return 0;
}

