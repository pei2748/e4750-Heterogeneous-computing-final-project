__global__ void ds(float *ds,int * y,  float *delta,  const unsigned int r, const unsigned int c )
{
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;
		
	   if(row < r && col < c)  {
		
		if( delta[row * c + col] > 0)
			ds[row * c + col ] = 1;
		__syncthreads();

		ds[row * c + y[row]] = 0;



	   } 
}
