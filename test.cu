#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define N 6

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

__global__ void reduce(float *a, int size, int index){
	int i;
        int tid=threadIdx.x;
	int start= ((index+tid+1)*size+index);
	int end= ((index+tid+1)*size+size);

        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
		 a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
        }

}


int main(){
	float a[(N*N)], c[(N*N)];
	float result[N][N],b[N][N];
	float *dev_a, *dev_b, *dev_c;
	int i;
	int j;
	int k;
	float l1;
	float u1;

        int threads=((N*N)-1);	
	//allocate the memory on the GPU
	cudaMalloc ( (void**)&dev_a, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_b, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_c, N*N* sizeof (float) );


	srand((unsigned)2);
	//fill the arrays 'a' and 'b' on the CPU
	for ( i = 0; i <= (N*N); i++) {
		a[i] =((rand()%10)+1);
	}
	
	printf("Matrix a is :\n");
	for(i=0; i<(N*N); i++){
           if(i%N==0)
          printf("\n %f ", a[i]);
           else printf("%lf ",a[i]);
         }

	cudaMemcpy( dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);//copy array to device memory
       
        /*Perform LU Decomposition*/
      	printf("\n=========================================================="); 
	for(i=0;i<N;i++){
        printf("\nPerforming scaling \n");
        scale<<<1,1>>>(dev_a,N,i);
      	reduce<<<1,(N-i-1)>>>(dev_a,N,i);

       /* 
        scale<<<1,1>>>(dev_a,N,0);//scaling step
        reduce<<<1, (N-0-1)>>>(dev_a, N, 0);//reduction step
	
	scale<<<1,1>>>(dev_a,N,1);
 	reduce<<<1,(N-1-1)>>>(dev_a,N,1);

	scale<<<1,1>>>(dev_a,N,2);
 	reduce<<<1,(N-2-1)>>>(dev_a,N,2);

	scale<<<1,1>>>(dev_a,N,3);
 	reduce<<<1,(N-3-1)>>>(dev_a,N,3);

	scale<<<1,1>>>(dev_a,N,4);
 	reduce<<<1,(N-4-1)>>>(dev_a,N,4);

	scale<<<1,1>>>(dev_a,N,5);
 	reduce<<<1,(N-5-1)>>>(dev_a,N,5);*/
       }
        /*LU decomposition ends here*/

	cudaMemcpy( c, dev_a, N*N*sizeof(float),cudaMemcpyDeviceToHost );//copy array back to host

       //display the results
	printf("\nCopied matrix C is \n");
	for ( i = 0; i < (N*N); i++) {
               if(i%N==0)
		printf( "\n%f  ", c[i]);  
               else printf("%lf ",c[i]);
	}
	printf("\n");	

	/*copy the result matrix into explicit 2D matrix for verification*/
        for(i=0;i<N;i++){
             for(j=0;j<N;j++){
		result[i][j]=c[i*N+j];
		}
        }

        /*printf("The result matrix\n");	
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


        printf("==================================================");
         printf("\nThe b matrix\n");	

         for(i=0;i<N;i++){
         	for(j=0;j<N;j++){
		printf("%lf ",b[i][j]);	
		}
	  printf("\n");
          }


	//free the memory allocated on the GPU
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	
	return 0;
}



