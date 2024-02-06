#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_SIZE 3
#define PADDING (KERNEL_SIZE / 2)
#define THREADS_PER_BLOCK 256

// Convolution kernel
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 0, 0},
    {-1, 1, 0},
    {0, 0, 0}
};

void checkCudaError(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

__global__ void convolutionZP_kernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int x = tid % width;
    int y = tid / width;

    if (x >= PADDING && x < width - PADDING && y >= PADDING && y < height - PADDING) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0;
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    int pixel_x = x + kx - PADDING;
                    int pixel_y = y + ky - PADDING;
                    sum += input[((pixel_y * width + pixel_x) * channels) + c] * kernel[ky][kx];
                }
            }
            output[((y * width + x) * channels) + c] = (unsigned char)sum;
        }
    }
}

void convolutionZP_CUDA(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int size = width * height * channels;
    unsigned char *d_input, *d_output;

    cudaMalloc((void **)&d_input, size * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, size * sizeof(unsigned char));

    cudaMemcpy(d_input, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    convolutionZP_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output, d_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    // Load image
    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    printf("Image loaded: %dx%d, %d channels\n", width, height, channels);

    if (image == NULL) {
        fprintf(stderr, "Error loading image %s\n", argv[1]);
        free(image);
        return -1;
    }

    // Allocate memory for the output image
    unsigned char *output = (unsigned char *)malloc(width * height * channels);

    // Call CUDA convolution function
    convolutionZP_CUDA(image, output, width, height, channels);

    stbi_write_png("results/bliss_conv_zp.png", width, height, channels, output, width * channels);
    printf("Output image saved to results/bliss_conv_zp.png\n");

    // Free memory
    stbi_image_free(image);
    free(output);

    return 0;
}
