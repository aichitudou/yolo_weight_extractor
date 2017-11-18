#include "im2col.h"
#include <stdio.h>
#include <time.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int channels_col = channels * ksize * ksize;
    char filename[50];
    sprintf(filename, "cjy/%d_im2col.txt", (int)time(NULL));
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);
    fprintf(stderr,"im2col index recording...");
    fprintf(fp,"c : %d\n",c);
    fprintf(fp,"h : %d\n",h);
    fprintf(fp,"w : %d\n",w);
    fprintf(fp,"height_col : %d\n",height_col);
    fprintf(fp,"width_col : %d\n",width_col);
    fprintf(fp,"pad : %d\n",pad);
    fprintf(fp,"channels_col : %d\n\n\n",channels_col);
    fflush(fp);

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);

                fprintf(fp,"c,h,w : %d, %d, %d\n",c,h,w);
                fprintf(fp,"w_offset, h_offset : %d, %d\n",w_offset,h_offset);
                fprintf(fp,"col_idx, im_row, im_col, c_im : %d, %d, %d, %d\n",col_index, im_row, im_col, c_im);
                fflush(fp);
		

		
		
            }
        }
    }
    fclose(fp);
}

