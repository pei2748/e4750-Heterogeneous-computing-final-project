#define TW 10

__global__ void x_dot_w(float *a, float *b, float *c, const unsigned int X, const unsigned int Y,  const unsigned int Z)
{		
	     int tx = threadIdx.x;
	     int ty = threadIdx.y;
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;

	      float temp = 0;
	      __shared__ float S_X [10][TW];
	      __shared__ float S_Y [10][TW];

	      for (int t = 0; t < (Y-1)/TW + 1; t++) {
			 if(row < X && (t* TW +tx) < Y )
				 S_X[ty][tx] = a[row * Y + t*TW + tx];
			 else 
				S_X[ty][tx] = 0.0;
			 
			 if ( (t* TW + ty)  < Y && col < Z ) 
				S_Y[ty][tx] = b[(t*TW + ty)* Z + col];
				
			  else 
				 S_Y[ty][tx] = 0.0;
			  
			 __syncthreads();

	    		for (int k = 0; k < TW; k++) {
		
	            		temp+= S_X[ty][k] * S_Y[k][tx];
			}
			__syncthreads();	
		}
	   if(row < X && col <Z) {
       		c[row * Z + col] = temp;
	   } 
}
