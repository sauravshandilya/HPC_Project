#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

//https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
#define TILE_DIM 32
#define BLOCK_ROWS 8

//http://cuda-programming.blogspot.in/2013/01/cuda-c-program-for-matrix-addition-and.html
#define TILE_WIDTH 2

// Image size .
#define ROW 32
#define COL 32

// Number of images
#define NUM 10

__global__ void
matrixSub( int *A, int* tosub)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < (ROW*COL*NUM))
    {
        A[i] = A[i] -(*tosub);
    }
}

__global__ void
matrixDiv( int *A, int todiv)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < (ROW*COL*NUM))
    {
        A[i] = A[i]/todiv;
    }
}


// taken from NVIDIA examples on effeicent matrix transpose

__global__ void transposeNaive(int *odata, int *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}


// to multiply. 
 __global__ void
MatrixMul( int *Md , int *Nd , int *Pd ,  int WIDTH )
{

           // calculate thread id

           unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;

           unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

         for (int k = 0 ; k<WIDTH ; k++ )
         {
                  Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
          }
}




int main (void){
    
    
    // read images and stuff. 
    
    // convert images to 1 D
    
    // all the images into one matrix.
    // Dimension of that matrix: (ROW*COL)xNUM
    
    int h_imageMatrix [ROW*COL][NUM]; // Ideal case, use malloc to allocate the memory.
    
    // Original case : Shud be filled with image data.
    // CUrrent Ideal case : copy this to DEVICE MEMORY and then initialise to random numbers b/w 0-255.
    // But rrandom seems buggy to me, so initing it with random numbers here on CPU.
    
    int max_number = 256;
    int minimum_number = 0;
    int i,j = 0;
    for(i=0; i < (ROW*COL); i++){
        for(j=0; j < NUM; j++){
            h_imageMatrix[i][j] = rand() % (max_number + 1 - minimum_number) + minimum_number;
        }
    }
    
    
    // Copy this data to DEVICE, since we will be working on this matrix there again.
    // No point for simply cpoying it for average
    
    int *d_imageMatrix, *d_imageMatrixT; // this form is used in GPU. No 2D array form
    int *d_average;        // this will contain the average.
    
    cudaError_t err = cudaSuccess;
    const size_t size = sizeof(int) * size_t(ROW*COL*NUM);
    
    // ALLOCATE memory for the ARRAY
    
    err = cudaMalloc((void **)&d_imageMatrix, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector IMAGE (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_imageMatrix, h_imageMatrix, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    
    // ALLOCATE MEM for AVERAGE
    err = cudaMalloc((void **)&d_average, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector IMAGE (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // ALLOCATE MEMORY for TRANSPOSE
    err = cudaMalloc((void **)&d_imageMatrixT, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector IMAGE (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    
    // ALLOCATE MEM for Multiplication of transpose and original
    // Dimension : ROW * ROW;
    
    int *d_matrixMul;
    
    err = cudaMalloc((void **)&d_matrixMul, size_t(ROW*ROW));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector IMAGE (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    
    
    
    // call the kernel for averaging.
    // but here we will be usign CPU for this. Not much imporvement in any case. 
    int h_sum = 0;
    for(i=0; i < (ROW*COL); i++){
        for(j=0; j < NUM; j++){
            h_sum += h_imageMatrix[i][j];
        }
    }
    
    int h_average = h_sum/(ROW*COL*NUM);
    
    
    // call the kernel for substracting
    
    // before that copy average to be subtsracted to the DEVICE
    

    err = cudaMemcpy(d_average, &h_average, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Please rehceck this. I am good at CUDA architecture.
    int threadsPerBlock = 256;
    int blocksPerGrid =((ROW*COL*NUM) + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    matrixSub<<<blocksPerGrid, threadsPerBlock>>>(d_imageMatrix, d_average);
    
    
    
    
    // call the kernel for transposing
    transposeNaive<<<blocksPerGrid, threadsPerBlock>>>(d_imageMatrixT, d_imageMatrix);
    
    // calculate covariance now
    
    // Multiply d_imageMatrix and its transpose
    
    MatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_imageMatrix, d_imageMatrixT, d_matrixMul, ROW);
    
    // divide all by M;
    

    
    
    

}