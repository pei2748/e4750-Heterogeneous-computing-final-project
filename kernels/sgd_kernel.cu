/*
 * parallelise for dot product and dw calculation.
 */

#define length_of_features 12
#define examples 455
 
__global__ void sgd(float *x, float* y, float* weights,
	   float reg_strength,	   
	   float learning_rate,
	   int total_examples)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	float val=0;
	float distance;
	int idx, itr;
	int data_point;
	__shared__ float dw[length_of_features];
	__shared__ float weights_shared[length_of_features];
	__shared__ float rand_index[examples];
	float dot_XW_single = 0;
	__shared__ float dot_XW;


	if (tid < length_of_features) {
		/* loading weights to shared memory*/
		weights_shared[tx] = weights[tid];
		// if block_size = 16, feature_len = 32, 
		__syncthreads();
		

		
		for(itr =0; itr < total_examples; itr++) {

			data_point = itr;
			
			/*  x[data_point] is a vector
			 *  each tid is computing one feature 
			 *  dot_XW_single = np.dot(X, W)
			 */

			idx = data_point * length_of_features + tid;
			dot_XW_single = x[idx] * weights_shared[tx];
			atomicAdd(&dot_XW, dot_XW_single);
			distance = 1 - (y[data_point] * dot_XW);

			if (distance > 0) {
				dw[tid] = weights_shared[tx]
					- (reg_strength * y[data_point] * x[idx]);
			} else
				dw[tid] = weights_shared[tx];

			val = learning_rate * dw[tx];
			weights_shared[tx] = weights_shared[tx] - val;
			__syncthreads();
		}//End--of--Data-Point
	
	__syncthreads();

	weights[tid] = weights_shared[tx];       

}//End--of--threadId-bound
}//End--of--global
