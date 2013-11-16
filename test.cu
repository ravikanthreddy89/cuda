#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define N 5

__global__ void add( float *a, float *b, float *c) {
	int tid = blockIdx.x;	//Handle the data at the index

		c[tid] = a[tid] + b[tid];
}


__global__ void scale(float *a, int size, int index){
 	int i;
	int start=(index*size);
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
	float a[(N*N)], b[(N*N)], c[(N*N)];
	float result[N][N];
	float *dev_a, *dev_b, *dev_c;
	int i;
	int j;
        int threads=((N*N)-1);	
	//allocate the memory on the GPU
	cudaMalloc ( (void**)&dev_a, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_b, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_c, N*N* sizeof (float) );
	
	//fill the arrays 'a' and 'b' on the CPU
	for ( i = 0; i <= (N*N); i++) {
		a[i] = i+1;
		b[i] = i+1;
	}
	
	printf("Vector a is :\n");
	for(i=0; i<(N*N); i++){
           if(i%N==0)
          printf("\n %f ", a[i]);
           else printf("%lf ",a[i]);
         }

       /*	printf("\nVector b is :\n");
       	for(i=0; i<(N*N); i++) {
           if(i%N==0)
           printf("\n%f  ", b[i]);
           else printf("%lf ",b[i]);
	 }*/
	//copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy( dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);
								
	//add<<<threads, 1>>> (dev_a, dev_b, dev_c);

        /*Perform LU Decomposition*/
        
	for(i=0;i<N;i++){
        scale<<<1,1>>>(dev_a,N,i);

        reduce<<<1, (N-0-1)>>>(dev_a, N, i);
          
        }

         /*LU decomposition ends here*/

	//copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( c, dev_a, N*N*sizeof(float),cudaMemcpyDeviceToHost );
								
								
	//display the results
	printf("\nVector c = a+b:\n");
	for ( i = 0; i < (N*N); i++) {
//		printf( "%d + %d = %d\n", a[i], b[i],c[i]);
               if(i%N==0)
		printf( "\n%f  ", c[i]);  
               else printf("%lf ",c[i]);
	}
	printf("\n");	

	/*Verification step */
        for(i=0;i<N;i++){
             for(j=0;j<N;j++){
		result[i][j]=c[i*N+j];
		}
        }
	

         for(i=0;i<N;i++){
         	for(j=0;j<N;j++){
		printf("%lf ",result[i][j]);	
		}
	  printf("\n");
          }
	//free the memory allocated on the GPU
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	
	return 0;
}



