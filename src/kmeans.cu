struct GlobalConstants {
    float3* points;
    int numPoints;
    int k;
    uint* clusters;
};

__constant__ GlobalConstants params;


__global__ void kernelComputeCluster(int numPoints, int k) {

    int modPoint = threadIdx.x;
    int point = blockIdx.x *1024 + threadIdx.x;
    int x = params.points[point].x;
    int y = params.points[point].y;
    int z = params.points[point].z;
    __shared__ uint prevCluster[1024];
    __shared__ uint cluster[1024];
    __shared__ float3 centers[k];
    __shared__ float totals[k];
    __shared__ int cont = 1;

    randomAssignment(centers);
    while (cont) {
        cont = 0
        min = 0
        minval =  (x-centers[0].x)^2 + (y-centers[0].y)^2 + (z-centers[0].z)^2;
        for(int i=1; i<k; i++) {
            float temp = (x-centers[i].x)^2 + (y-centers[i].y)^2 + (z-centers[i].z)^2;
            if (temp < minval) {
                min = i;
                minval = temp;
            }
        }
        cluster[point] = min;
        if (prevCluster[point] != cluster[point]) {
            cont = 1;
        }
        __syncthreads();

        if(point < 3*k) {
            ((float*) centers)[point] = 0;
            if (point%3==0) totals[point/k] = 0;
            for(int i=0; i<1024; i++) {
                if (cluster[i]==point/k) {
                    centers[point] += ((float*)params.points)[i*3+(k%3)]
                    if (point%3==0) totals[point/k] += 1;
                }

            }
            __syncthreads();
            centers[point] /= totals[point/k]
        }
        prevCluster = cluster;
    }
    params.clusters = cluster;
}

void
kmeans::kmeans() {

    dim3 blockDim(1024);
    dim3 gridDim((params.numPoints + 1023)/1024);

    kernelComputeCluster<<<gridDim,blockDim>>>(params.numPoints, params.k);
    cudaDeviceSynchronize();
    
}
