#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<sys/time.h>

//#define N 6

__global__ void add( float *a, float *b, float *c) {
	int tid = blockIdx.x;	//Handle the data at the index

		c[tid] = a[tid] + b[tid];
}


__global__ void scale(float *a, int size, int index){
 	int i;
	int start=(index*size+index);
	int end=(index*size+size);
	
	for(i=start+1;i<end;i++){
		a[i]=(a[i]/a[start]);
	}

}

__global__ void reduce(float *a, int size, int index, int b_size){
	extern __shared__ float pivot[];
	int i;

	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int block_size=b_size;

	int pivot_start=(index*size+index);
	int pivot_end=(index*size+size);

	int start;
	int end;
	int pivot_row;
	int my_row;

	if(tid==0){
	     for(i=index;i<size;i++) pivot[i]=a[(index*size)+i];
	}

	__syncthreads();

	pivot_row=(index*size);
	my_row=(((block_size*bid) + tid)*size);
	start=my_row+index;
	end=my_row+size;

	if(my_row >pivot_row){
        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
		// a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
		 a[i]=a[i]-(a[start]*pivot[(i-my_row)]);

             }
        }

}


int main(int argc, char *argv[]){
	float *a;
	float *c;
	float error;
	int N;
	int flag=0;	

 	float **result;
	float **b;
	int blocks;
        
        float *dev_a;
	int i;
	int j;
	int k;
	float l1;
	float u1;

	double start;
	double end;
	struct timeval tv;
	
	N=atoi(argv[1]);	
	//allocate memory on CPU
	a=(float *)malloc(sizeof(float)*N*N);
	c=(float *)malloc(sizeof(float)*N*N);


	result=(float **)malloc(sizeof(float *)*N);
	b=(float **)malloc(sizeof(float *)*N);


	for(i=0;i<N;i++){
	   result[i]=(float *)malloc(sizeof(float)*N);
   	   b[i]=(float *)malloc(sizeof(float)*N);
	}

	//allocate the memory on the GPU
	cudaMalloc ( (void**)&dev_a, N*N* sizeof (float) );

	srand((unsigned)2);
	//fill the arrays 'a' on the CPU
	for ( i = 0; i <= (N*N); i++) {
		a[i] =((rand()%10)+1);
	}
	
	printf("Matrix a is :\n");
	/*for(i=0; i<(N*N); i++){
           if(i%N==0)
          printf("\n %f ", a[i]);
           else printf("%lf ",a[i]);
         }*/

	cudaMemcpy(dev_a,a,N*N*sizeof(float), cudaMemcpyHostToDevice);//copy array to device memory
       
 	gettimeofday(&tv,NULL);
	start=tv.tv_sec;
        /*Perform LU Decomposition*/
      	printf("\n==========================================================\n"); 
	for(i=0;i<N;i++){
         scale<<<1,1>>>(dev_a,N,i);
	// blocks= ((N-i-1)/512)+1;
	blocks=((N/512));
//	printf("Number of blocks rxd : %d \n",blocks);
	reduce<<<blocks,512,N*sizeof(float)>>>(dev_a,N,i,512);
 
       }
        /*LU decomposition ends here*/

 	gettimeofday(&tv,NULL);
	end=tv.tv_sec;
	cudaMemcpy( c, dev_a, N*N*sizeof(float),cudaMemcpyDeviceToHost );//copy array back to host

	printf("\nThe time for LU decomposition is %lf \n",(end-start));
       //display the results
//	printf("\nCopied matrix C is \n");
/*	for ( i = 0; i < (N*N); i++) {
               if(i%N==0)
		printf( "\n%f  ", c[i]);  
               else  printf("%lf ",c[i]);
	}*/
	printf("\n");	

	/*copy the result matrix into explicit 2D matrix for verification*/
       for(i=0;i<N;i++){
             for(j=0;j<N;j++){
		result[i][j]=c[i*N+j];
		}
        }

    /*    printf("The result matrix\n");	
        for(i=0;i<N;i++){
         	for(j=0;j<N;j++){
		printf("%lf ",result[i][j]);	
		}
	  printf("\n");
          }*/

	printf("=======================================================");
	printf("\n Performing inplace verification \n");
        /*Inplace verification step*/

       for(i=0;i<N;i++){
           for(j=0;j<N;j++){
                b[i][j]=0;
              for(k=0;k<N;k++){
                 if(i>=k)l1=result[i][k];
                  else l1=0;

                  if(k==j)u1=1;
                  else if(k<j)u1=result[k][j];//figured it out 
                  else u1=0.0;

               b[i][j]=b[i][j]+(l1*u1);

             }
           }
         }


	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
		error=abs(a[(i*N+j)]-b[i][j]);
		if(error> 1 ) { 
	        //	printf("No match occured at %d %d Error is %lf \n ", i, j, abs(a[(i*N+j)]-b[i][j]));
                        flag =flag+1;
                       }	
	       }
	}

	if(flag==0) printf("Match");
	else printf("No Matchs %d \n",flag);
        printf("\n==================================================\n");
        // printf("\nThe b matrix\n");	
         
        /*for(i=0;i<N;i++){
         	for(j=0;j<N;j++){
		printf("%lf ",b[i][j]);	
		}
	  printf("\n");
          }*/

        
	//free the memory allocated on the GPU
	cudaFree( dev_a );
	
	return 0;
}



