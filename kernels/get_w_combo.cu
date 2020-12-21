__global__ void get_w_combo(float *a,float*b,  float *w,  const unsigned int r, const unsigned int Y ,const unsigned int c )
{
	     int col =  blockDim.x * blockIdx.x + threadIdx.x;
	     int row =  blockDim.y * blockIdx.y + threadIdx.y;


	     if(row < r && col <c) {
	        float temp = 0;

	        for (int k = 0; k < Y; k++) {
		     temp+= a[row * Y + k] * b[k * c + col];
		}

		
		float temp1= w[row * c + col];
		float temp3 = 2 * 50000 * temp1;
	     	float temp4 = (temp/200)  + temp3;
		float temp5 = 0.0000001 * temp4;	
		   
		w[row * c + col] = temp1 - temp5;



	   } 
}
