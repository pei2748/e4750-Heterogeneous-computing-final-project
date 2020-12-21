#define length_of_features 12 

__global__ void  sgd_lock_free_naive(float *x, float* y, float* weights,
	   	 			   float reg_strength,
					   float learning_rate,
					   int total_examples,
					   int max_epochs)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float val=0;
	float dw[length_of_features];
	float temp=0;
	float distance;
	int idx;

	if(tid < total_examples) {
		for(int i=0 ; i < max_epochs; i++){
				idx = tid * length_of_features;
				for(int k=0; k < length_of_features; k++)
					temp += x[idx + k] * weights[k];

				distance = 1 - (y[tid] * temp);

				for (int ii=0; ii< length_of_features ; ii++) {
					if (distance > 0) {
						dw[ii] = weights[ii]
							- (reg_strength * y[tid] * x[idx + ii]);
					} else
						dw[ii] = weights[ii];
				}

				for (int jj=0; jj< length_of_features ; jj++) {
					val = learning_rate * dw[jj];
					weights[jj]= weights[jj] -val;
				}//End--of--weight--update

				__syncthreads(); 
		}//End--of--for-epoch    
	}//End--of--if-bound
}//End--of--global
