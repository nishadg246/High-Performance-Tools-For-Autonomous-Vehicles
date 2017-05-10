#include <stdio.h>
#include <stdlib.h>
#include <math.h>       /* sqrt */

#define PI (3.14159265)

int main(){
	int32_t num = 1000000;
	float *data = (float*)malloc(num*sizeof(float));

	// pointers
	float *px = data+0;
	float *py = data+1;
	float *pz = data+2;
	float *pr = data+3;

	// load point cloud
	FILE *stream;
	stream = fopen ("../../2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000117.bin","rb");
	if (stream==NULL) {
		perror("ERROR: Cannot open lidar file");
		exit(EXIT_FAILURE);
	}

	num = fread(data,sizeof(float),num,stream)/4;








	FILE *imageFile;
	int x,y,pixel,height=500,width=1000;
	//int x,y,pixel,height=3,width=2;

	imageFile=fopen("image.ppm","wb");
	if(imageFile==NULL){
		perror("ERROR: Cannot open output file");
		exit(EXIT_FAILURE);
	}

	fprintf(imageFile,"P6\n");               // P6 filetype
	fprintf(imageFile,"%d %d\n",width,height);   // dimensions
	fprintf(imageFile,"255\n");              // Max pixel

	unsigned char pix[(width*height)*3];
	
	for(int i = 0; i < ((width*height)*3); i++){
		pix[i] = 0;
	}

	for(int i = 0; i < num; i+=3){
		
		float fval = (sqrt(((*px)*(*px)) + ((*py)*(*py)) + ((*pz)*(*pz))))/100.0;
		int val = 255- ((int)((fval)*255));

		float rad = (PI + atan2(*py,*px));
		int x = (int)((rad/(2.0*PI))*width);
		float t1 = (*pz);
		float t2 = (((5.0+t1)/10.0)*height);
		int z = height - ((int)t2);
		int index = ((z * width) + x)*3;	

		px+=4; py+=4; pz+=4; pr+=4;

		//printf("(%f)->(%d,%d,%d, %f,%f) -> [%d,%d,%d]\n",(*pz),index,x,z,t1,t2,val,val,val);

		pix[index] = val;
		pix[index + 1] = val;
		pix[index + 2] = val;
		
		//printf("%f -> %d -> %d\n",fval,temp,val);
		
	}
	

	fwrite(pix,1,width * height * 3,imageFile);
	fclose(stream);
	fclose(imageFile);
}