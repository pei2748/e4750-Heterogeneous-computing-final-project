#define TW 10
__global__ void get_w_combo(float *a,float*b,  float *w,  const unsigned int r, const unsigned int Y ,const unsigned int c )
{		
	     int tx = threadIdx.x;
	     int ty = threadIdx.y;
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;

	     float temp = 0;
	      __shared__ float S_X [10][TW];
	      __shared__ float S_Y [10][TW];

	     for (int t = 0; t < (Y-1)/TW + 1; t++) {
	              if(row < r && (t* TW +tx) < Y )
	                     S_X[ty][tx] = a[row * Y + t*TW + tx];
	              else
	                     S_X[ty][tx] = 0.0;
                      if ( (t* TW + ty)  < Y && col < c )
                             S_Y[ty][tx] = b[(t*TW + ty)* c + col];
                      else
                             S_Y[ty][tx] = 0.0;
																                             __syncthreads();
                      for (int k = 0; k < TW; k++) {
                             temp+= S_X[ty][k] * S_Y[k][tx];
																                             }
                      __syncthreads();
              }

	     if(row < r && col <c) {

		float temp1= w[row * c + col];
		float temp3 = 2 * 50000 * temp1;
	     	float temp4 = (temp/200)  + temp3;
		float temp5 = 0.0000001 * temp4;	
		   
		w[row * c + col] = temp1 - temp5;



	   } 
}
