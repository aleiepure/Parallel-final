#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#define KERNEL_SIZE 3
// #define BLOCK_SIZE 3 * 10

__constant__ float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}
};

__global__ void convolutionKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0;
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    int pixel_x = col + kx - KERNEL_SIZE / 2;
                    int pixel_y = row + ky - KERNEL_SIZE / 2;

                    if (pixel_x >= 0 && pixel_x < width && pixel_y >= 0 && pixel_y < height) {
                        sum += input[(pixel_y * width + pixel_x) * channels + c] * kernel[ky][kx];
                    }
                }
            }
            // Normalize sum to range 0-255
            sum = sum < 0 ? 0 : sum;
            sum = sum > 255 ? 255 : sum;
            output[(row * width + col) * channels + c] = (unsigned char)sum;
        }
    }
}

int main(int argc, char **argv) {
    // Check command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path to image>\n", argv[0]);
        return -1;
    }

    // Load image
    int width, height, channels;
    int blocks_num = 10;
    
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    int block_size = channels * blocks_num;
    
    printf("Image loaded: %dx%d, %d channels\n", width, height, channels);

    if (image == NULL) {
        fprintf(stderr, "Error loading image %s\n", argv[1]);
        return -1;
    }

    // Allocate memory for the output image
    unsigned char *output = (unsigned char *)malloc(width * height * channels);

    // Copy kernel to constant memory
    //cudaMemcpyToSymbol(constant_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, width * height * channels * sizeof(unsigned char));

    // Copy input image to device memory
    cudaMemcpy(d_input, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch convolution kernel
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    // Copy result back to host
    cudaMemcpy(output, d_output, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image
    stbi_write_png("results/bliss_conv_cuda.png", width, height, channels, output, width * channels);
    printf("Output image saved to results/bliss_conv_cuda.png\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    stbi_image_free(image);
    free(output);

    return 0;
}
