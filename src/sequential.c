/*
 * Intro to Parallel Computing, Final Project
 * A.Y. 2023/2024
 * Authors: Alessandro Iepure, 228023
 *          Lorenzo Fasol, 227561
 *          Riccardo Minella, 227326
 *
 * Sequential implementation of the image processing algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#define KERNEL_SIZE 3

// Convolution kernel
// const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
//     {-1, -1, -1},
//     {-1,  8, -1},
//     {-1, -1, -1}
// };

const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 1, 0},
    {1, 4, 1},
    {0, 1, 0}
};

// const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
//     {0, 0, 0},
//     {0, 1, 0},
//     {0, 0, 0}
// };

// const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
//     {1.0 / 16, 1.0 / 8, 1.0 / 16},
//     {1.0 / 8, 1.0 / 4, 1.0 / 8},
//     {1.0 / 16, 1.0 / 8, 1.0 / 16}
// };

// Convolution without zero padding
void convolution(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int pixel_x = x + kx - 1;
                        int pixel_y = y + ky - 1;
                        sum += input[((pixel_y * width + pixel_x) * channels) + c] * kernel[ky][kx];
                    }
                }
                output[((y * width + x) * channels) + c] = (unsigned char)sum;
            }
        }
    }
}

// Convolution with zero padding
void convolutionZP(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    // Create a padded version of the input image
    int padded_width = width + 2 * (KERNEL_SIZE - 1);
    int padded_height = height + 2 * (KERNEL_SIZE - 1);
    unsigned char *padded_input = (unsigned char *)malloc(padded_width * padded_height * channels);

    // Initialize padded_input with zeros
    memset(padded_input, 0, padded_width * padded_height * channels);

    // Copy the original image to the center of the padded image
    for (int y = 2; y < height + (KERNEL_SIZE - 1); y++) {
        for (int x = 2; x < width + (KERNEL_SIZE - 1); x++) {
            for (int c = 0; c < channels; c++) {
                padded_input[(y * padded_width + x) * channels + c] = input[((y - (KERNEL_SIZE - 1)) * width + (x - (KERNEL_SIZE - 1))) * channels + c];
            }
        }
    }

    stbi_write_png("../prova.png", padded_width, padded_height, channels, padded_input, padded_width * channels);

    // Perform convolution on the padded image
    for (int y = 1; y < padded_height - 1; y++) {
        for (int x = 1; x < padded_width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int pixel_x = x + kx - 1;
                        int pixel_y = y + ky - 1;
                        sum += padded_input[((pixel_y * padded_width + pixel_x) * channels) + c] * kernel[ky][kx];
                    }
                }
                output[((y - 1) * padded_width + (x - 1)) * channels + c] = (unsigned char)sum;
            }
        }
    }

    // Free memory
    free(padded_input);
}


int main() {
    // Load image
    int width, height, channels;
    unsigned char *image = stbi_load("../bliss.png", &width, &height, &channels, 0);

    if (image == NULL) {
        fprintf(stderr, "Error loading image\n");
        return -1;
    }

    // Allocate memory for the output image
    unsigned char *output = (unsigned char *)malloc(width * height * channels);
    unsigned char *outputZP = (unsigned char *)malloc((width + 2 * (KERNEL_SIZE - 1)) * (height + 2 * (KERNEL_SIZE - 1)) * channels);

    // Apply convolution
    convolution(image, output, width, height, channels);
    convolutionZP(image, outputZP, width, height, channels);

    // Save the result
    stbi_write_png("../results/bliss_conv.png", width, height, channels, output, width * channels);
    printf("Output image saved to results/bliss_conv.png\n");

    stbi_write_png("../results/bliss_conv_zp.png", width + 2 * (KERNEL_SIZE - 1), height + 2 * (KERNEL_SIZE - 1), channels, outputZP, (width + 2 * (KERNEL_SIZE - 1)) * channels);
    printf("Output image saved to results/bliss_conv_zp.png\n");

    // for (int i = 0; i < width * height * channels; i++) {
    //     if (image[i] != output[i])
    //         printf("Input[%d] = %d, Output[%d] = %d\n", i, image[i], i, output[i]);
    // }

    // Free memory
    stbi_image_free(image);
    free(output);

    return 0;
}