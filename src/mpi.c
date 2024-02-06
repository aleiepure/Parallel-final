// https://github.com/0xnirmal/Parallel-Convolution-MPI

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#define KERNEL_SIZE 3

const int PADDING = KERNEL_SIZE / 2;

// Convolution kernel
const int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0, 1, 0}, {1, -4, 1}, {0, 1, 0}};

void print_matrix(int rank, unsigned char *matrix, int height, int width,
                  int channels) {
  printf("\nRank %d:\n", rank);
  for (int i = 0; i < height; i++) {
    printf("\t");
    for (int j = 0; j < width; j++)
      for (int c = 0; c < channels; c++)
        printf("%003d ", matrix[i * width * channels + j * channels + c]);
    printf("\n");
  }
}

int main(int argc, char **argv) {

  // MPI Initialization
  MPI_Init(&argc, &argv);

  int MPI_rank, MPI_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
  MPI_Status status;
  MPI_Datatype chunk_type;

  // Parse arguments
  if (argc != 2) {
    if (MPI_rank == 0)
      printf("Usage: %s <input_file>\n", argv[0]);

    MPI_Finalize();
    return 1;
  }

  // Load image
  int width, height, channels;
  int paddedWidth, paddedHeight;
  int rowsPerProcess;
  unsigned char *image, *paddedImage;

  if (MPI_rank == 0) {

    // Load image
    image = stbi_load(argv[1], &width, &height, &channels, 0);
    printf("Image loaded: %dx%d, %d channels\n", width, height, channels);

    // width = 1600;
    // height = 1600;
    // channels = 3;
    // unsigned char *image = (unsigned char *)malloc(width * height * channels
    // *
    //                                                sizeof(unsigned char));
    // for (int i = 0; i < height; i++)
    //   for (int j = 0; j < width; j++)
    //     image[i * width + j] = 255;
    // print_matrix(-1, image, height, width);

    // Zero-pad image
    paddedWidth = width + 2 * PADDING;
    paddedHeight = height + 2 * PADDING;

    paddedImage = (unsigned char *)malloc(paddedWidth * paddedHeight *
                                          channels * sizeof(unsigned char));

    memset(paddedImage, 0, paddedWidth * paddedHeight * channels);
    for (int y = PADDING; y < height + PADDING; y++) {
      for (int x = PADDING; x < width + PADDING; x++) {
        for (int c = 0; c < channels; c++) {
          paddedImage[(y * paddedWidth + x) * channels + c] =
              image[((y - PADDING) * width + (x - PADDING)) * channels + c];
        }
      }
    }

    rowsPerProcess = height / MPI_size;
  }

  MPI_Bcast(&paddedWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&paddedHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rowsPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Type_vector(rowsPerProcess + 2 * PADDING, paddedWidth * channels,
                  paddedWidth * channels, MPI_UNSIGNED_CHAR, &chunk_type);
  MPI_Type_commit(&chunk_type);

  if (MPI_rank == 0) {
    for (int i = 0; i < MPI_size; i++) {
      MPI_Send(paddedImage + (paddedWidth * rowsPerProcess * channels * i), 1,
               chunk_type, i, 0, MPI_COMM_WORLD);
    }

    stbi_image_free(image);
    free(paddedImage);
  }

  unsigned char *localImage =
      (unsigned char *)malloc(paddedWidth * (rowsPerProcess + 2 * PADDING) *
                              channels * sizeof(unsigned char));

  MPI_Recv(localImage, 1, chunk_type, 0, 0, MPI_COMM_WORLD, &status);
  print_matrix(MPI_rank, localImage, rowsPerProcess + 2 * PADDING, paddedWidth,
               channels);

  unsigned char *localConvolutedImage = (unsigned char *)malloc(
      width * rowsPerProcess * channels * sizeof(unsigned char));

  unsigned char *localPaddedConvolutedImage =
      (unsigned char *)malloc(paddedWidth * (rowsPerProcess + 2 * PADDING) *
                              channels * sizeof(unsigned char));

  // Convolution
  for (int y = PADDING; y < rowsPerProcess + PADDING; y++) {
    for (int x = PADDING; x < paddedWidth - PADDING; x++) {
      for (int c = 0; c < channels; c++) {
        if (channels == 4 && c == 3)
          localPaddedConvolutedImage[((y * paddedWidth + x) * channels) + c] =
              localImage[((y * paddedWidth + x) * channels) + c];
        else {
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
          if (sum < 0)
            localPaddedConvolutedImage[((y * paddedWidth + x) * channels) + c] =
                0;
          else if (sum > 255)
            localPaddedConvolutedImage[((y * paddedWidth + x) * channels) + c] =
                255;
          else
            localPaddedConvolutedImage[((y * paddedWidth + x) * channels) + c] =
                (unsigned char)sum;
        }
      }
    }
  }
  for (int y = 0; y < rowsPerProcess; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        localConvolutedImage[(y * width + x) * channels + c] =
            localPaddedConvolutedImage
                [((y + PADDING) * paddedWidth + (x + PADDING)) * channels + c];
      }
    }
  }

  MPI_Send(localConvolutedImage, 1, chunk_type, 0, 0, MPI_COMM_WORLD);

  if (MPI_rank == 0) {
    unsigned char *outputImage = (unsigned char *)malloc(
        width * height * channels * sizeof(unsigned char));

    for (int i = 0; i < MPI_size; i++) {
      MPI_Recv(outputImage + (i * rowsPerProcess * width * channels), 1,
               chunk_type, i, 0, MPI_COMM_WORLD, &status);
    }
    stbi_write_png("results/mpi.png", width, height, channels, outputImage, 0);
    print_matrix(-1, outputImage, height, width, channels);
    free(outputImage);
  }

  free(localImage);
  free(localConvolutedImage);
  free(localPaddedConvolutedImage);

  // Compute the rows that need to be added to each chunk. The first and last
  // will result to be zero-padded.
  // unsigned char upper[paddedWidth];
  // unsigned char lower[paddedWidth];

  // unsigned char *paddingRowUpper = NULL;
  // unsigned char *paddingRowLower = NULL;

  // int rowsPerProcess = height / MPI_size;
  // int startRow = rowsPerProcess * MPI_rank;
  // int endRow = rowsPerProcess - 1 + startRow;
  // int next = (MPI_rank + 1) % MPI_size;
  // int prev = (MPI_rank != 0) ? (MPI_rank - 1) : (MPI_size - 1);
  // int numRows = endRow + 1 - startRow;

  // memcpy(lower, &hPaddedImage[paddedWidth * (endRow - PADDING + 1)],
  //        sizeof(unsigned char) * paddedWidth * channels);
  // paddingRowLower = malloc(sizeof(unsigned char) * paddedWidth);

  // memcpy(upper, &hPaddedImage[paddedWidth * startRow],
  //        sizeof(unsigned char) * paddedWidth * channels);
  // paddingRowUpper = malloc(sizeof(unsigned char) * paddedWidth);

  // // Create the padded local images
  // // Top row of rank 0 and bottom row of last rank are filled with zeros,
  // other
  // // ranks are computed with the values sent/received from the neighbors
  // if (MPI_size > 1) {

  //   if (MPI_rank % 2 == 1) {
  //     MPI_Recv(paddingRowLower, paddedWidth, MPI_UNSIGNED_CHAR, next, 1,
  //              MPI_COMM_WORLD, &status);
  //     MPI_Recv(paddingRowUpper, paddedWidth, MPI_UNSIGNED_CHAR, prev, 1,
  //              MPI_COMM_WORLD, &status);
  //   } else {
  //     MPI_Send(upper, paddedWidth, MPI_UNSIGNED_CHAR, prev, 1,
  //     MPI_COMM_WORLD); MPI_Send(lower, paddedWidth, MPI_UNSIGNED_CHAR, next,
  //     1, MPI_COMM_WORLD);
  //   }

  //   if (MPI_rank % 2 == 1) {
  //     MPI_Send(upper, paddedWidth, MPI_UNSIGNED_CHAR, prev, 0,
  //     MPI_COMM_WORLD); MPI_Send(lower, paddedWidth, MPI_UNSIGNED_CHAR, next,
  //     0, MPI_COMM_WORLD);
  //   } else {
  //     MPI_Recv(paddingRowLower, paddedWidth, MPI_UNSIGNED_CHAR, next, 0,
  //              MPI_COMM_WORLD, &status);
  //     MPI_Recv(paddingRowUpper, paddedWidth, MPI_UNSIGNED_CHAR, prev, 0,
  //              MPI_COMM_WORLD, &status);
  //   }

  // } else {
  //   paddingRowLower = upper;
  //   paddingRowUpper = lower;
  // }

  // unsigned char *localImage = malloc(sizeof(unsigned char) * paddedWidth *
  //                                    (numRows + (2 * PADDING)) * channels);

  // if (MPI_rank == 0) {
  //   memset(paddingRowUpper, 0,
  //          paddedWidth * sizeof(unsigned char) * 2 * PADDING * channels);
  // }
  // if (MPI_rank == (MPI_size - 1)) {
  //   memset(paddingRowLower, 0,
  //          paddedWidth * sizeof(unsigned char) * 2 * PADDING * channels);
  // }
  // memcpy(localImage, paddingRowUpper,
  //        sizeof(unsigned char) * paddedWidth * PADDING * channels);

  // memcpy(localImage + (height * PADDING),
  //        hPaddedImage + (numRows + (2 * PADDING) * startRow),
  //        sizeof(unsigned char) * numRows + (2 * PADDING) * numRows);

  // memcpy(localImage + (height * (numRows + PADDING)), paddingRowLower,
  //        sizeof(unsigned char) * numRows + (2 * PADDING) * PADDING);

  // memcpy(localImage, paddingRowUpper,
  //        sizeof(unsigned char) * paddedWidth * PADDING); // Top row
  // memcpy(localImage + height * PADDING, image + paddedWidth * startRow,
  //        sizeof(unsigned char) * height * numRows); // Middle rows
  // memcpy(localImage + height * (PADDING + numRows), paddingRowLower,
  //        sizeof(unsigned char) * paddedWidth * PADDING + 1); // Bottom row
  // MPI_Barrier(MPI_COMM_WORLD);
  // print_matrix(MPI_rank, localImage, numRows + (2 * PADDING), paddedWidth);

  // for (int y = PADDING; y < paddedHeight - PADDING; y++) {
  //   for (int x = PADDING; x < paddedWidth - PADDING; x++) {
  //     for (int c = 0; c < channels; c++) {

  //       // Skip the alpha channel if the image has one
  //       if (channels == 4 && c == 3)
  //         convolutedLocalImage[((y * paddedWidth + x) * channels) + c] =
  //             localImage[((y * paddedWidth + x) * channels) + c];

  //       else {
  //         float sum = 0.0;
  //         for (int ky = 0; ky < KERNEL_SIZE; ky++) {
  //           for (int kx = 0; kx < KERNEL_SIZE; kx++) {

  //             if (y + ky - PADDING < 0 || y + ky - PADDING >= paddedHeight ||
  //                 x + kx - PADDING < 0 || x + kx - PADDING >= paddedWidth) {
  //               sum = convolutedLocalImage[((y * paddedWidth + x) * channels)
  //               +
  //                                          c] = -1;
  //             } else {
  //               int pixel_x = x + kx - PADDING;
  //               int pixel_y = y + ky - PADDING;
  //               sum +=
  //                   localImage[((pixel_y * paddedWidth + pixel_x) * channels)
  //                   +
  //                              c] *
  //                   kernel[ky][kx];
  //             }
  //           }
  //         }
  //         convolutedLocalImage[((y * paddedWidth + x) * channels) + c] =
  //             (unsigned char)sum;
  //       }
  //     }
  //   }
  // }
  // print_matrix(convolutedLocalImage, (endRow - startRow), width);

  // // Gather the local chunks to the root process
  // MPI_Gather(outputImage, (endRow - startRow) * IMAGE_WIDTH, MPI_INT,
  //              imageData, (endRow - startRow) * IMAGE_WIDTH, MPI_INT,
  //              0, MPI_COMM_WORLD);

  // Save output image

  // Clean up
  // free(paddingRowLower);
  // free(paddingRowUpper);
  // free(localImage);
  // free(convolutedLocalImage);
  MPI_Finalize();

  return 0;
}
