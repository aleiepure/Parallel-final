/*
 * Intro to Parallel Computing, Final Project
 * A.Y. 2023/2024
 * Authors: Alessandro Iepure, 228023
 *          Lorenzo Fasol, 227561
 *          Riccardo Minella, 227326
 *
 * MPI implementation of the image processing algorithm.
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

// Constants
#define KERNEL_SIZE 3
const int PADDING = KERNEL_SIZE / 2;
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 1, 0}, {1, -4, 1}, {0, 1, 0}};

/**
 * Main function
 *
 * @param argc The number of arguments
 * @param argv The arguments
 * @return The exit code
 */
int main(int argc, char **argv) {

  // Parse arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);

    return 1;
  }

  // MPI Initialization
  MPI_Init(&argc, &argv);

  int MPI_rank, MPI_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

  // Variables
  int width = 0, height = 0, channels = 0;
  int paddedWidth = 0, paddedHeight = 0;
  int rowsPerProcess = 0;
  unsigned char *image = NULL, *paddedImage = NULL, *localImage = NULL,
                *localConvolutedImage = NULL,
                *localPaddedConvolutedImage = NULL, *outputImage = NULL;
  int sendcounts[MPI_size];
  int displs[MPI_size];
  int next_rank = MPI_rank + 1;
  int prev_rank = MPI_rank - 1;
  double start = 0.0, end = 0.0;

  // Load the image info
  if (access(argv[1], F_OK) == -1) {
    fprintf(stderr, "Image file does not exist: %s\n", argv[1]);

    // Clean up
    MPI_Finalize();

    return -1;
  }

  stbi_info(argv[1], &width, &height, &channels);
  if (MPI_rank == 0)
    printf("Image info: %dx%d, %d channels\n", width, height, channels);

  if (height == 0 || width == 0 || channels == 0) {
    if (MPI_rank == 0)
      fprintf(stderr, "Failed to load image info\n");

    MPI_Finalize();
    return -1;
  }

  if (height % MPI_size != 0) {
    if (MPI_rank == 0)
      fprintf(stderr, "The image height must be divisible by the number of "
                      "processes\n");

    // Clean up
    MPI_Finalize();

    return -1;
  }

  // Rank 0 loads the image and pads it
  if (MPI_rank == 0) {
    image = stbi_load(argv[1], &width, &height, &channels, 0);

    start = MPI_Wtime();

    // Zero-pad image horizontally
    paddedWidth = width + 2 * PADDING;
    paddedHeight = height;
    paddedImage = (unsigned char *)malloc(paddedWidth * paddedHeight *
                                          channels * sizeof(unsigned char));
    memset(paddedImage, 0, paddedWidth * paddedHeight * channels);

    for (int y = 0; y < height; y++) {
      for (int x = PADDING; x < width + PADDING; x++) {
        for (int c = 0; c < channels; c++) {
          paddedImage[(y * paddedWidth + x) * channels + c] =
              image[(y * width + (x - PADDING)) * channels + c];
        }
      }
    }
  }

  // Broadcast the image dimensions
  MPI_Bcast(&paddedWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Create the chunk type
  rowsPerProcess = height / MPI_size;
  MPI_Status status;
  MPI_Datatype chunk_type;
  MPI_Type_vector(rowsPerProcess, paddedWidth * channels,
                  paddedWidth * channels, MPI_UNSIGNED_CHAR, &chunk_type);
  MPI_Type_commit(&chunk_type);

  // Send the chunks to the processes
  localImage =
      (unsigned char *)malloc(sizeof(unsigned char) * paddedWidth *
                              (rowsPerProcess + 2 * PADDING) * channels);
  memset(localImage, 0,
         sizeof(unsigned char) * paddedWidth * (rowsPerProcess + 2 * PADDING) *
             channels);

  for (int i = 0; i < MPI_size; i++) {
    sendcounts[i] = 1;
    displs[i] = i;
  }
  MPI_Scatterv(paddedImage, sendcounts, displs, chunk_type,
               &localImage[paddedWidth * channels], 1, chunk_type, 0,
               MPI_COMM_WORLD);

  // Send/receive the first and last rows to/from the neighboring processes
  if (MPI_rank == 0) {
    MPI_Send(&localImage[paddedWidth * channels * rowsPerProcess],
             paddedWidth * channels, MPI_UNSIGNED_CHAR, next_rank, 0,
             MPI_COMM_WORLD);

    MPI_Recv(&localImage[paddedWidth * channels * (rowsPerProcess + 1)],
             paddedWidth * channels, MPI_UNSIGNED_CHAR, next_rank, 0,
             MPI_COMM_WORLD, &status);
  } else if (MPI_rank == (MPI_size - 1)) {
    MPI_Send(&localImage[paddedWidth * channels], paddedWidth * channels,
             MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD);

    MPI_Recv(&localImage[0], paddedWidth * channels, MPI_UNSIGNED_CHAR,
             prev_rank, 0, MPI_COMM_WORLD, &status);
  } else {
    MPI_Send(&localImage[paddedWidth * channels * rowsPerProcess],
             paddedWidth * channels, MPI_UNSIGNED_CHAR, next_rank, 0,
             MPI_COMM_WORLD);

    MPI_Recv(&localImage[0], paddedWidth * channels, MPI_UNSIGNED_CHAR,
             prev_rank, 0, MPI_COMM_WORLD, &status);

    MPI_Send(&localImage[paddedWidth * channels], paddedWidth * channels,
             MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD);

    MPI_Recv(&localImage[paddedWidth * channels * (rowsPerProcess + 1)],
             paddedWidth * channels, MPI_UNSIGNED_CHAR, next_rank, 0,
             MPI_COMM_WORLD, &status);
  }

  // Apply convolution locally
  localPaddedConvolutedImage =
      (unsigned char *)malloc(paddedWidth * (rowsPerProcess + 2 * PADDING) *
                              channels * sizeof(unsigned char));
  localConvolutedImage = (unsigned char *)malloc(
      width * rowsPerProcess * channels * sizeof(unsigned char));

  for (int y = PADDING; y < rowsPerProcess + PADDING; y++) {
    for (int x = PADDING; x < paddedWidth - PADDING; x++) {
      for (int c = 0; c < channels; c++) {

        // Ignore alpha channel if present
        if (channels == 4 && c == 3) {
          localPaddedConvolutedImage[((y * paddedWidth + x) * channels) + c] =
              localImage[((y * paddedWidth + x) * channels) + c];
        } else {
          float sum = 0.0;
          for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
              int pixel_x = x + kx - PADDING;
              int pixel_y = y + ky - PADDING;
              sum += localImage[((pixel_y * paddedWidth + pixel_x) * channels) +
                                c] *
                     kernel[ky][kx];
            }
          }

          // Clamp the result to [0, 255]
          sum = sum < 0 ? 0 : sum;
          sum = sum > 255 ? 255 : sum;
          localPaddedConvolutedImage[((y * paddedWidth + x) * channels) + c] =
              (unsigned char)sum;
        }
      }
    }
  }

  // Remove padding from result
  for (int y = 0; y < rowsPerProcess; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        localConvolutedImage[(y * width + x) * channels + c] =
            localPaddedConvolutedImage
                [((y + PADDING) * paddedWidth + (x + PADDING)) * channels + c];
      }
    }
  }

  // Send the convoluted chunks back to the root process
  if (MPI_rank == 0)
    outputImage = (unsigned char *)malloc(width * height * channels *
                                          sizeof(unsigned char));

  for (int i = 0; i < MPI_size; i++) {
    sendcounts[i] = width * channels * rowsPerProcess;
    displs[i] = i * rowsPerProcess * width * channels;
  }
  MPI_Gatherv(localConvolutedImage, width * channels * rowsPerProcess,
              MPI_UNSIGNED_CHAR, outputImage, sendcounts, displs,
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  // Save output image
  if (MPI_rank == 0) {
    end = MPI_Wtime();
    printf("Elapsed time: %f ms\n", (end - start) * 1000.0);

    char filename[100];
    sprintf(filename, "results/mpi_%d.png", MPI_size);
    stbi_write_png(filename, width, height, channels, outputImage, 0);
    printf("Output image saved to %s.\n", filename);
  }

  // Clean up
  if (MPI_rank == 0) {
    stbi_image_free(image);
    free(outputImage);
    free(paddedImage);
  }
  free(localImage);
  free(localConvolutedImage);
  free(localPaddedConvolutedImage);

  MPI_Type_free(&chunk_type);
  MPI_Finalize();

  return 0;
}
