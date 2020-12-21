#include <stdio.h>
#include <stdlib.h>


#define LEN_F 3073
#define TILE_WIDTH 32

// 3073/32 = 97.

__global__ void sgd(float *x, float* y, float* weights,
		    float *single_dw, /* dw computed by one data point, with size (3073, 10) */
		    float reg_strength,	   
		    float learning_rate,
		    int total_examples,
		    float *dot_XW,
		    float *loss) /* dot_XW is with size (10, 1) */
{
        /* blockDim.x = 10, blockDim.y = 32 */
	int tx = threadIdx.x; //10
	int ty = threadIdx.y; //32

	float tmp_w, tmp_dw;
	int yi, t, data_point;

	__shared__ float weights_shared[TILE_WIDTH][10];
	__shared__ float x_shared[TILE_WIDTH];
	__shared__ float ds[10];
	__shared__ float sum_ds;
	__shared__ float distance[10];
	__shared__ float loss_i[10];
	__shared__ float W_square;
	__shared__ float sum_loss;
	float W_square_single = 0;	

        float sum_value=0;
	// 2D block, (10, 32, 1)
	if (tx == 0 && ty ==0) {
	  sum_loss = 0;
	  W_square = 0;
	}
	for(data_point =0; data_point < total_examples; data_point++) {

	       for (t = 0; t < (LEN_F-1)/TILE_WIDTH + 1; t++) {
			if ((t * TILE_WIDTH + ty) < LEN_F) 
				weights_shared[ty][tx] = weights[(t * TILE_WIDTH + ty)* 10 + tx];
			else
				weights_shared[ty][tx] =0;
			if( (t*TILE_WIDTH+ty) < LEN_F)
				x_shared[ty]  = x[data_point * LEN_F + t *TILE_WIDTH + ty];
			else
				x_shared[ty]=0;
			__syncthreads();

			for(int k=0 ; k < TILE_WIDTH; k++)
			  sum_value+= x_shared[k] * weights_shared[k][tx];
	       }//end--of--tile		
	       // tx is the indexing of column {0, 1, 2, ..., 9}

	       dot_XW[tx] = sum_value;		
	       __syncthreads();
		// dot_XW should finish updating by all threads
		
	       if(ty==0) {
		 yi = (int) y[data_point]; //6
		 distance[tx] = dot_XW[tx] -  dot_XW[yi] + 1;
	       }
	       __syncthreads();
		
	       if(ty==0) { 
		 if (distance[tx] > 0) {
		   ds[tx] = 1;
		 }  else {
		   ds[tx] = 0;
		 }
		 ds[yi] = 0;
		 atomicAdd(&sum_ds, ds[tx]);
	       }      
	       __syncthreads();		

	       // calculating loss by accumulating 200 data point.
	       if(ty==0) {
	         if (distance[tx] > 0) {
		   loss_i[tx] = distance[tx];
		 }  else {
		   loss_i[tx] = 0;
		 }
		 loss_i[yi] = 0;
		 atomicAdd(&sum_loss, loss_i[tx]); //loss_i is (10, 1)		 
//		 __syncthreads();
	       }      
	       
	       
	       if(ty==0) {
		 ds[yi] = -1 * sum_ds;
		 for(int ii=0 ; ii< LEN_F ; ii++) {
		   int idx = ii * 10 + tx;
		   single_dw[idx] += x[ii] * ds[tx];
		 }    
	       } 
	       __syncthreads(); // wait for all 10 threads to finish the single_dw matrix.
	}//End--of--Data-Point	   
	if(ty==0) {
	  for(int ii=0 ; ii< LEN_F ; ii++) {
	    int idx = ii* 10 + tx;	    
	    tmp_w = weights[idx];	    
	    tmp_dw = single_dw[idx]/total_examples  + 2 * reg_strength * tmp_w;
	    W_square_single += tmp_w * tmp_w; // calculate for one column of W, for computing loss	    
	    weights[idx] = tmp_w -  learning_rate * tmp_dw;
	  }
	  // 10 threads add to W_square;
	  atomicAdd(&W_square, W_square_single);
//	  __syncthreads();
	  if (tx == 0) { // only one thread is calculating the loss
	    loss[0] = sum_loss/total_examples + reg_strength * W_square;
	  }
	}//end--of--ty 
	__syncthreads();

}//End--of--global
