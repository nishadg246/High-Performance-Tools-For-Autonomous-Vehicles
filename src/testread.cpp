#include <stdio.h>
#include <stdlib.h>

int main() {
  int32_t num = 1000000;
  float *data = (float*)malloc(num*sizeof(float));

  // pointers
  float *px = data+0;
  float *py = data+1;
  float *pz = data+2;
  float *pr = data+3;
  int count = 0;

  // load point cloud
  FILE *stream;
  stream = fopen ("0000000000.bin","rb");
  uint pix[100000][100000][100000] = new uint[100000][100000][100000];
  if (stream!=NULL) {
    num = fread(data,sizeof(float),num,stream)/4;
    for (int32_t i=0; i<num; i++) {
      
      px+=4; py+=4; pz+=4; pr+=4;

    }
    printf("%d\n", num-count);
    fclose(stream);
  }


  FILE *imageFile;
  imageFile=fopen("image.ppm","wb");
  if(imageFile==NULL){
    perror("ERROR: Cannot open output file");
    exit(EXIT_FAILURE);
  }
  fprintf(imageFile,"P6\n");               // P6 filetype
  fprintf(imageFile,"%d %d\n",1000,1000);   // dimensions
  fprintf(imageFile,"255\n");              // Max pixel
  fwrite(pix,1,1000*1000*3,imageFile);
  fclose(imageFile);
}
