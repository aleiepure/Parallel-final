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

const int PADDING = KERNEL_SIZE / 2;

// Convolution kernel
// const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
//     {0, 0, 0},
//     {1, -1, 0},
//     {0, 0, 0}
// };

const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}
};

// Convolution with zero padding
void convolutionZP(unsigned char *input, unsigned char *outputZP, int width, int height, int channels) {
    double start = clock();

    // Create a padded version of the input image
    int padded_width = width + 2 * PADDING;
    int padded_height = height + 2 * PADDING;
    unsigned char *padded_input = (unsigned char *)malloc(padded_width * padded_height * channels);
    unsigned char *padded_output = (unsigned char *)malloc(padded_width * padded_height * channels);

    // Initialize padded_input with zeros
    memset(padded_input, 0, padded_width * padded_height * channels);

    // Copy the original image to the center of the padded image
    for (int y = PADDING; y < height + PADDING; y++) {
        for (int x = PADDING; x < width + PADDING; x++) {
            for (int c = 0; c < channels; c++) {
                padded_input[(y * padded_width + x) * channels + c] = input[((y - PADDING) * width + (x - PADDING)) * channels + c];
            }
        }
    }

    for (int y = PADDING; y < padded_height - PADDING; y++) {
        for (int x = PADDING; x < padded_width - PADDING; x++) {
            for (int c = 0; c < channels; c++) {
                if (channels == 4 && c == 3)
                    padded_output[((y * padded_width + x) * channels) + c] = padded_input[((y * padded_width + x) * channels) + c];
                else {
                    float sum = 0.0;
                    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                            int pixel_x = x + kx - PADDING;
                            int pixel_y = y + ky - PADDING;
                            sum += padded_input[((pixel_y * padded_width + pixel_x) * channels) + c] * kernel[ky][kx];
                        }
                    } 
                    if (sum < 0)
                        padded_output[((y * padded_width + x) * channels) + c] = 0;
                    else if (sum > 255)
                        padded_output[((y * padded_width + x) * channels) + c] = 255;
                    else
                    padded_output[((y * padded_width + x) * channels) + c] = (unsigned char)sum;
                }
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                outputZP[(y * width + x) * channels + c] = padded_output[((y + PADDING) * padded_width + (x + PADDING)) * channels + c];
            }
        }
    }

    double end = clock();
    printf("Convolution time: %f\n", (end - start) / CLOCKS_PER_SEC);

    // Free memory
    free(padded_input);
    free(padded_output);
}

int main(int argc, char **argv) {

    // Check command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path to image>\n", argv[0]);
        return -1;
    }

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

    convolutionZP(image, output, width, height, channels);

    stbi_write_png("results/seq.png", width, height, channels, output, width * channels);
    printf("Output image saved to results/seq.png\n");

    // Free memory
    stbi_image_free(image);
    free(output);

    return 0;
}