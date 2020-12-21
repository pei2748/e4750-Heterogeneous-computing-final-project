/*
 * parallelise for dot product and dw calculation.
 */

#define length_of_features 12
#define examples 455
#define TILE_WIDTH 64
 
__global__ void sgd(float *x, float* y, float* weights,
	   float reg_strength,	   
	   float learning_rate,
	   int total_examples)
	 
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float val=0;
	float distance;
	int idx, itr;
	int data_point;
	__shared__ float dw[length_of_features];
	__shared__ float weights_shared[length_of_features];
    __shared__ float x_shared[TILE_WIDTH][length_of_features];
    __shared__ float y_shared[TILE_WIDTH];
	float dot_XW_single = 0;
	__shared__ float dot_XW;
	float temp_weight;
	float temp_x;
    int tile_bound = (examples -1)/TILE_WIDTH +1;

	if (tx < length_of_features) {
		/* loading weights to shared memory*/
		weights_shared[tx] = weights[tid];
		// if block_size = 16, feature_len = 32, 
		__syncthreads();
		
    
       
        for (int t =0 ; t < tile_bound ; t++) {
             	int s_index = t* TILE_WIDTH + ty;

            if(s_index  < examples) {
                x_shared[ty][tx] = x[s_index * length_of_features + tx ];
                y_shared[ty]     = y[s_index ];
            }
            else {
                x_shared[ty][tx] = 0;
                y_shared[ty] =0;
            }
       
    	   if(ty==0) {	    
            for(data_point =0; data_point < TILE_WIDTH; data_point++) {       
                
                /*  x[data_point] is a vector
                    *  each tid is computing one feature 
                    *  dot_XW_single = np.dot(X, W)
                    */

                idx = data_point  ;
		temp_weight = weights_shared[tx];
		temp_x = x_shared[idx][tx];

                dot_XW_single = temp_x * temp_weight;
                atomicAdd(&dot_XW, dot_XW_single);
                distance = 1 - (y_shared[idx] * dot_XW);

                if (distance > 0) {
                    dw[tid] = temp_weight
                        - (reg_strength * y_shared[idx] * temp_x);
                } else
                    dw[tid] = temp_weight;

                val = learning_rate * dw[tx];
                weights_shared[tx] = temp_weight - val;
            }//End--of--Data-Point
	}//end--of--ty
            __syncthreads();
        }//end--of-tile    

    weights[tid] = weights_shared[tx];       

    }//End--of--threadId-bound
}//End--of--global
