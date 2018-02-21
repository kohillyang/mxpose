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
		int32_t center_x,
		int32_t center_y,
		double * output){

//	printf("%d %d %d %d\r\n",height,width,center_x,center_y);
	for(int i = max(0,center_y-25);i<min(height,center_y+25);i++){
		for(int j = max(0,center_x-25);j<min(width,center_x+25);j++){
			int index = i*width + j;
			int d2 =( i-center_y)*( i-center_y) + (j - center_x)*(j - center_x);
			output[index] = exp(-1.0*d2/7/7/2);
		}
	}
	return;
}




