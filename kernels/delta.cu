__global__ void delta(float *s, float *y, float *delta, const unsigned int r, const unsigned int c )
{
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;
		
	   if(row < r && col < c)  {
		
		delta[row * c + col] = s [row * c + col] - y[row] +1;



	   } 
}
