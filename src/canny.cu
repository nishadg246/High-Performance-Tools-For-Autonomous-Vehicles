struct GlobalConstants {
    float3* points;
    float* gaussianMask;
    float* gradXMask;
    float* gradYMask;
    uint* clusters;
};

__constant__ GlobalConstants params;


__global__ void kernelGaussanFilter(float* filtered, float* image) {

    float tot = 0;
    int row = blockDim.y*blockIdx.y +  threadIdx.y;
    int col = blockDim.x*blockIdx.x +  threadIdx.x;
    for(int i=0;i<5;i++) {
        for(int j=0; j<5;j++) {
            tot += gaussianMask[i*5+j] * image[row+i][col+j];
        }
    }
    filtered[row][col] = tot

}

__global__ void kernelGradients(float* gradient, float* angle, float image) {

    float gradX = 0;
    float gradY = 0;
    int row = blockDim.y*blockIdx.y +  threadIdx.y;
    int col = blockDim.x*blockIdx.x +  threadIdx.x;
    for(int i=-1;i<2;i++) {
        for(int j=-1; j<2;j++) {
            gradX += gradXMask[(i+1)*3+j+1] * image[row+i][col+j];
            gradY += gradYMask[(i+1)*3+j+1] * image[row+i][col+j];
        }
    }
    gradient[row][col] = sqrt(gradX^2+gradY^2);
    angle[row][col] = arctan(gradY/gradX);

}


void
kmeans::kmeans() {

    dim3 blockDim(32,32);
    dim3 gridDim(width/32, height/32);

    kernelGaussanFilter<<<gridDim,blockDim>>>(filtered, image);
    cudaDeviceSynchronize();

    kernelGradients<<<gridDim,blockDim>>>(gradient, angle, image);
    cudaDeviceSynchronize();
    
}
