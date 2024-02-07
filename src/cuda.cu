#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#define KERNEL_SIZE 3
#define PADDING (KERNEL_SIZE / 2)

__constant__ float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}
};

__global__ void paddingKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int c = 0; c < channels; c++) {
            output[((row + PADDING) * (width + 2*PADDING) + (col + PADDING)) * channels + c] = input[(row * width + col) * channels + c];
        }
    }
}

__global__ void unpaddingKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int c = 0; c < channels; c++) {
            output[(row * width + col) * channels + c] = input[((row + PADDING) * (width + 2*PADDING) + col + PADDING) * channels + c];
        }
    }
}

__global__ void convolutionKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int padded_width = width + 2 * PADDING;
    int padded_height = height + 2 * PADDING;

    if (row < padded_height && col < padded_width) {
        for (int c = 0; c < channels; c++) {
            if (channels == 4 && c == 3)
                output[(row * padded_width + col) * channels + c] = input[(row * padded_width + col) * channels + c];
            else {
                float sum = 0.0;
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        int pixel_x = col + kx - PADDING;
                        int pixel_y = row + ky - PADDING;

                        if (pixel_x >= 0 && pixel_x < padded_width && pixel_y >= 0 && pixel_y < padded_height) {
                            sum += input[(pixel_y * padded_width + pixel_x) * channels + c] * kernel[ky][kx];
                        }
                    }
                }
                // Normalize sum to range 0-255
                sum = sum < 0 ? 0 : sum;
                sum = sum > 255 ? 255 : sum;
                output[(row * padded_width + col) * channels + c] = (unsigned char)sum;
            }
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

    // Allocate device memory
    unsigned char *d_input, *d_output, *d_padded_output, *d_padded_input;
    cudaMalloc((void **)&d_input, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_padded_output, (width + 2*PADDING) * (height + 2*PADDING) * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_padded_input, (width + 2*PADDING) * (height + 2*PADDING) * channels * sizeof(unsigned char));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

    // Copy input image to device memory
    cudaMemcpy(d_input, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for original image
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch padding kernel
    paddingKernel<<<gridSize, blockSize>>>(d_input, d_padded_input, width, height, channels);

    // Define grid and block dimensions for padded image
    dim3 gridSizePadded((width + 2*PADDING + blockSize.x - 1) / blockSize.x, (height + 2*PADDING + blockSize.y - 1) / blockSize.y);

    // Launch convolution kernel
    convolutionKernel<<<gridSizePadded, blockSize>>>(d_padded_input, d_padded_output, width, height, channels);

    // Launch unpadding kernel
    unpaddingKernel<<<gridSize, blockSize>>>(d_padded_output, d_output, width, height, channels);

    // Copy result back to host
    cudaMemcpy(output, d_output, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Convolution time: %f ms\n", milliseconds);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Save output image
    stbi_write_png("results/cuda.png", width, height, channels, output, width * channels);
    printf("Output image saved to results/cuda.png\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_padded_input);
    cudaFree(d_padded_output);

    // Free host memory
    stbi_image_free(image);
    free(output);

    return 0;
}
