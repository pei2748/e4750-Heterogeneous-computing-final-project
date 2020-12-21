__global__ void xT(float *a, float *b, const unsigned int X, const unsigned int Y)
{
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;

	   if(row < Y && col < X) {
	   

		b[row * X + col] = a[col * Y + row];
	   } 
}
