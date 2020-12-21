#include <stdio.h>
#include <stdlib.h>
#define LEN_F 3073
#define TILE_WIDTH 32
// 3073/32 = 97.

__global__ void sgd(float *x, float* y, float* weights,
		    float *loss,
		    float reg_strength,	   
		    float learning_rate,
		    int total_examples,
		    int max_epoch)
{
        /* blockDim.x = 200 */
	int tx = threadIdx.x; //200
	int tid_x = blockIdx.x * blockDim.x + tx; // col

	int yi;
	int idx, idx_w;
//	__shared__ float dw[LEN_F][10];
//	float reg_strength = 5e4;
//	float learning_rate = 1e-7;
	float ds[10];
	float distance[10];
	float dot_XW[10]; // stored in private mem of GPU
	float loss_i[10];

	float val, dot_tmp;
	float tmp_dw;
        float tmpW;
	float sum_ds;
        float sum_value;
	float W_square;
	float sum_loss;
	//__shared__ float mega_dw;

//	printf("before for loop start   ");

	// each tid takes one data point
	if (tid_x < total_examples) {
          /* calculate dot(W, x[data_point]) */
	  yi = (int) y[tid_x]; //6	

	  for(int epoch=0 ; epoch <max_epoch ; epoch++) {
	    // compute dot(x, W)
	    W_square = 0;
	    sum_loss = 0;
	    for (int w_col=0 ; w_col < 10 ; w_col++) {
	      sum_value = 0;	      
	      for(int k=0; k < LEN_F; k++) {
		idx = tid_x * LEN_F + k;
		idx_w = k * 10 + w_col;
		dot_tmp = x[idx] * weights[idx_w];		
		sum_value += dot_tmp;
//		if (w_col == 0 && tid_x == 10 && k == 0){
//		  printf("idx=%d ", idx); printf("idx_w=%d ", idx_w); printf("x[idx]=%f ", x[idx]);
//		  printf("weights[idx_w]=%f ", weights[idx_w]);printf("dot_tmp=%f ", dot_tmp);		 
//		}
//		
	      }//end of for-loop of features
	      dot_XW[w_col] = sum_value;
//	      if (tid_x == 10 && w_col == 0) {
//		printf("A: epoch = %d, sum_value = %f;    ", epoch, w_col, sum_value);
//		printf("A: weights[4096] = %f", weights[4096]);
//	      }
//	      
	    } // end-for-w_col, dot-product finished.

	    //	    __syncthreads();	      
	    // dot_XW should finish updating by all threads now.
	    //	    if (tid_x == 10)
	    //	      printf("dot_XW[0] = %f ", dot_XW[0]);
	    //
//	    if (tid_x == 10){
//	      printf("after syncthreads, weights[tid] = %f, weights[%d]= %f ;", weights[tid_x], tid_x, weights[10]);
//	    }
//	    
	    for(int d =0 ; d< 10 ; d++) {
	      distance[d] = dot_XW[d] -  dot_XW[yi] + 1;
	    }	

	    // calculate ds
	    // step 1: if corresponding distance > 0, set as 1.
	    //         if corresponding distance <= 0, set as 0.
	    for(int d =0; d< 10 ; d++) {
	      if (distance[d] > 0)
		ds[d] = 1.0f;
	      else 
		ds[d] = 0.0f;
	    }
	    // only calculate loss at the max_epoch
	    if (epoch == max_epoch - 1) {
	      for(int d =0; d< 10 ; d++) {
		if (distance[d] > 0)
		  loss_i[d] = distance[d];
		else
		  loss_i[d] = 0;
	      }
	      loss_i[yi] = 0;
	      for (int d=0; d< 10; d++)
		sum_loss += loss_i[d];
	    }
	    	    
	    // If yi = 5, set ds[5] = 0;
	    ds[yi] = 0;
	    sum_ds = 0;
	    // sum up 10 ds to sum_ds;
	    for (int d=0; d< 10; d++) {
	      sum_ds+= ds[d];
	    }
	      // __syncthreads();
	      // set ds[5] as -sum_ds 
	    ds[yi] = -1 * sum_ds;
//	    if (tid_x == 10){
//	      printf("In middle, weights[0] = %f ;", weights[0]);
//	    }
//	      
	    
	    for(int f=0 ; f < LEN_F ; f++) {
	      for (int c=0 ; c < 10; c++) {
		idx = f * 10 + c; 
		tmp_dw = x[tid_x * LEN_F + f] * ds[c];
		tmpW = weights[idx];
//		if (tid_x == 10 && f==0 && c == 0) {
//		  printf("x[%d]= %f, ds[%d]= %f ;", tid_x * LEN_F + f, x[tid_x * LEN_F + f], c, ds[c]);
//		  printf("weights[%d] = %f ;", idx, weights[idx]);		  
//		}
//		  
		if (epoch == max_epoch - 1)
		  W_square += tmpW * tmpW;			  
		if (tmp_dw != 0) {
		  val = learning_rate * tmp_dw + 2 * reg_strength * tmpW;
		  weights[idx] = tmpW - val;
		}
	      }
	    } // end-of-for-features	    
	    __syncthreads();
//	    if (tid_x == 10) {
//	      printf("epoch-%d, weights[0] = %f;  ", epoch, weights[0]);
//	    }
//	    
	  }//End--of--epoch
	  sum_loss += reg_strength * W_square;
//	  if (tid_x == 10) {
//	    	  printf("W_square[%d] = %f;  ", tid_x, W_square);		
////	  	  printf("loss[%d] = %f;  ", tid_x, sum_loss);
//	  }
//
	  loss[tid_x] = sum_loss;
          __syncthreads();
    }// end of if
}//End--of--global
