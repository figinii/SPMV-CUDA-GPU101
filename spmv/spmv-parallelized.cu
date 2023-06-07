#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

# define THREADS_PER_BLOCK 1024

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int *row_ptr_t = (int *) malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *) malloc(*num_vals * sizeof(int));
    float *values_t = (float *) malloc(*num_vals * sizeof(float));
    
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *) malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++) {
        row_occurances[i] = 0;
    }
    
    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        
        row_occurances[row]++;
    }
    
    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++) {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);
    
    // Set the file position to the beginning of the file
    rewind(file);
    
    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++) {
        col_ind_t[i] = -1;
    }
    
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }
    
    fclose(file);
    
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
}

// CPU implementation of SPMV using CSR
void spmv_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    for (int i = 0; i < num_rows; i++) {
        float dotProduct = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += values[j] * x[col_ind[j]];
        }
        
        y[i] = dotProduct;
    }
}

// function to check the CUDAMalloc allocation of mamory in GPU
#define CHECK(call){                                                                    \
    const cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                           \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}                                                                                       \

// function to ckeck the call to the Kernel CUDA to be executed on GPU
#define CHECK_KERNELCALL(){                                                             \
    const cudaError_t err = cudaGetLastError();                                         \
    if (err != cudaSuccess) {                                                           \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

// kernel to multiply the values in the matrix and in the vector
__global__ void spmv_csr_mul(const int *col_ind, float *values, const float *vec, int num_vals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    if(i<num_vals)
        values[i] = vec[col_ind[i]] * values[i];
}

// kernel to inizilise a vector
__global__ void vector_inizialiser(float *vector, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<dim)
        vector[i] = 0;
}

// kernel to sum the rows of the matrix
__global__ void parallel_sum(float *input, float *output, int dim_slice, int out_index, int start_slice){    
    __shared__ float shared_input[THREADS_PER_BLOCK * sizeof(float)];
    int t_id = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < dim_slice){
        shared_input[t_id] = input[index + start_slice];
        __syncthreads();

        for (unsigned int j=1; j < dim_slice; j *= 2) {
            int i = 2 * j * t_id;
            if (i < blockDim.x) {
                shared_input[i] += shared_input[i + j];
            }
            __syncthreads();
            }

        if (t_id == 0){
            output[out_index] += shared_input[0];
        }
    }
}

int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("Usage: ./exec matrix_file\n");
        return 0;
    }
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    
    const char *filename = argv[1];

    double start_cpu, end_cpu;
    
    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
    
    float *x = (float *) malloc(num_rows * sizeof(float));
    float *y_sw = (float *) malloc(num_rows * sizeof(float));

    // Generate a random vector

    srand(time(NULL));

    for (int i = 0; i < num_rows; i++) {
        x[i] = (rand()%100)/(rand()%100+1); //the number we use to divide cannot be 0, that's the reason of the +1 
    }

    // Compute in sw
    start_cpu = get_time();
    spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
    end_cpu = get_time();

    // Decleare GPU var for CSR

    int *col_gpu;
    float *values_gpu;
    double start_gpu, end_gpu, gpu_time;

    CHECK(cudaMalloc((int**)&col_gpu, num_vals * sizeof(float)));
    CHECK(cudaMalloc((float**)&values_gpu, num_vals * sizeof(float)));

    CHECK(cudaMemcpy(col_gpu, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(values_gpu, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    
    // Declear GPU var for input and output vector
    float *x_gpu, *y_gpu;

    CHECK(cudaMalloc((float**)&x_gpu, num_rows * sizeof(float)));
    CHECK(cudaMalloc((float**)&y_gpu, num_rows * sizeof(float)));
    
    CHECK(cudaMemcpy(x_gpu, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    
    /* ----- Execution on GPU ----- */

    dim3 blockPerGrid((num_vals/THREADS_PER_BLOCK) + 1,1,1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK,1,1);

    
    // FIRST Kernel for moltiplications
    start_gpu = get_time();
    spmv_csr_mul <<<blockPerGrid,threadsPerBlock>>>(col_gpu, values_gpu, x_gpu, num_vals);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    end_gpu = get_time();
    gpu_time = end_gpu - start_gpu;


    // SECOND Kernel for sum parallel reduction

    // inizialise the result vector
    blockPerGrid.x = (num_rows * sizeof(float) / THREADS_PER_BLOCK) + 1;
    start_gpu = get_time();
    vector_inizialiser <<<blockPerGrid,threadsPerBlock>>>(y_gpu, num_rows * sizeof(float));
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    end_gpu = get_time();
    gpu_time += end_gpu - start_gpu;

    // parallel sum
    int slice_start, slice_end, slice_dim;

    for(int i = 0; i < num_rows + 1; i++){
        slice_start = row_ptr[i];
        slice_end = row_ptr[i+1];
        slice_dim = slice_end - slice_start;
        blockPerGrid.x = ((slice_dim/THREADS_PER_BLOCK) + 1);
        start_gpu = get_time();
        parallel_sum<<<blockPerGrid,threadsPerBlock>>>(values_gpu, y_gpu, slice_dim, i, slice_start);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
        end_gpu = get_time();
        gpu_time += end_gpu - start_gpu;
    }
    
    float *y_hw = (float *) malloc(num_rows * sizeof(float));
    CHECK(cudaMemcpy(y_hw, y_gpu, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // check differences
    float diff = 0;

    for(int i = 0; i < num_rows; i++)
        diff = y_sw[i] - y_hw[i];   
    
    printf("tot difference: %d\n", diff);

    // Print time
    printf("SPMV Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SPMV Time GPU: %.10lf\n", gpu_time);

    // Free    
    free(row_ptr);
    free(col_ind);
    free(values);
    free(y_sw);

    CHECK(cudaFree(col_gpu));
    CHECK(cudaFree(values_gpu));
    CHECK(cudaFree(x_gpu));
    CHECK(cudaFree(y_gpu));

    return 0;
}
