__global__ void x_dot_w(float *a, float *b, float *c, const unsigned int X, const unsigned int Y,  const unsigned int Z)
{
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;

	   if(row < X && col <Z) {
	        float temp = 0;

	    for (int k = 0; k < Y; k++) {
	            temp+= a[row * Y + k] * b[k * Z + col];
	       }

       c[row * Z + col] = temp;
	   } 
}
