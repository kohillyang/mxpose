/*
 * heatpaf.c
 *
 *  Created on: Mar 9, 2018
 *      Author: kohill
 */
/*
 * heatmap.c
 *
 *  Created on: Feb 19, 2018
 *      Author: kohill
 */
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>
using namespace std;
//static int32_t max(int32_t x,int32_t y){
//    if (x>y)
//        return x;
//    else
//        return y;
//}
//static int32_t min(int32_t x,int32_t y){
//    if (x<y)
//        return x;
//    else
//        return y;
//}

extern "C" void gen_pafmap(float *keypoints,
		int32_t height,int32_t width,
		float *output1,float *output2,
		float *count,
		int32_t buffer_size,
		float stride = 8.0){
	assert(height * width == buffer_size);
	float x0 = keypoints[0];
	float y0 = keypoints[1];
	float x1 = keypoints[2];
	float y1 = keypoints[3];
	float norm = sqrt((y1-y0) * (y1-y0) + (x1-x0) * (x1-x0)) + 1e-3;
//	cout << x0 << y0 << x1 << y1 << endl;
	int start_n = (int)round((min(x1,x0) + 0.5 -  1.0*stride / 2) / stride) -2 ;
	int start_m = (int)round((min(y1,y0) + 0.5 -  1.0*stride / 2) / stride) -2 ;
	int end_n = (int)round((max(x1,x0) + 0.5 -  1.0*stride / 2) / stride) + 2 ;
	int end_m = (int)round((max(y1,y0) + 0.5 -  1.0*stride / 2) / stride) + 2 ;
	float sigma = 10.0;
//	cout << end_m << end_n << start_m << start_n;
	for(int m = max(start_m,0);m<min(height,end_m);m++){
		for (int n = max(start_n,0);n < min(width,end_n);n ++){
			int32_t index = m * width + n;
			float ori_x = n * stride + 1.0*stride / 2 -0.5;
			float ori_y = m * stride + 1.0*stride / 2 -0.5;
		    float distance = abs((y1-y0)*(ori_x-x0) + (y0-ori_y)*(x1-x0));
		    distance /= sqrt((y1-y0)*(y1-y0) + (x1-x0)*(x1-x0)) + 0.0001;
		    float exp_v = exp(-1 * distance * distance / (2 * sigma * sigma));
		    output1[index] = (output1[index] * count[index] + exp_v * (x1-x0))/(count[index] + 1)/norm;
		    output2[index] = (output1[index] * count[index] + exp_v * (y1-y0))/(count[index] + 1) /norm;
		    count[index] += 1;
		}
	}
}



