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
static int32_t max(int32_t x,int32_t y){
    if (x>y)
        return x;
    else
        return y;
}
static int32_t min(int32_t x,int32_t y){
    if (x<y)
        return x;
    else
        return y;
}
void genGaussionHeatmap(
		int32_t height,
		int32_t width,
		float center_x,
		float center_y,
		int32_t stride,
		double * output){

//	printf("%d %d %f %f\r\n",height,width,center_x,center_y);
	for(int i = max(0,center_y/stride-25);i<min(height,center_y/stride+25);i++){
		for(int j = max(0,center_x/stride-25);j<min(width,center_x/stride+25);j++){
//		    printf("%d %d",i,j);
		    double ori_x = j * stride + 1.0*stride / 2 -0.5;
		    double ori_y = i * stride + 1.0*stride / 2 -0.5;
			int index = i*width + j;
			double d2 =( ori_y-center_y)*( ori_y-center_y) + (ori_x - center_x)*(ori_x - center_x);
			output[index] = exp(-1.0*d2/7/7/2);
		}
	}
	return;
}




