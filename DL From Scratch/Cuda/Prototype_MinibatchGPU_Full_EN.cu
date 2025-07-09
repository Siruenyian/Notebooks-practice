// ============ MNIST Classifier from Scratch using GPU Acceleration ============
// This CUDA (.cu) file demonstrates the use of GPU acceleration for training a neural network.
// It applies a simple classifier to the MNIST dataset, leveraging CUDA kernels to perform
// key operations such as matrix multiplication, vector addition, and matrix transpotition efficiently on the GPU.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Loop to echeck Cuda availability
#define CUDA_CHECK(x)                                                                        \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = x;                                                                 \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

// Fucntion to print Cuda Array on device
void CopyAndPrintDeviceArray(const float *device_array, int batchSize, int numClasses, const char *label)
{
    int total = batchSize * numClasses;
    float *host_array = (float *)malloc(total * sizeof(float));
    if (!host_array)
    {
        fprintf(stderr, "Failed to allocate host memory for %s\n", label);
        return;
    }

    cudaError_t err = cudaMemcpy(host_array, device_array, total * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA memcpy failed for %s: %s\n", label, cudaGetErrorString(err));
        free(host_array);
        return;
    }

    printf("=== %s (batchSize = %d, numClasses = %d) ===\n", label, batchSize, numClasses);
    for (int b = 0; b < batchSize; ++b)
    {
        printf("Batch %d: ", b);
        for (int c = 0; c < numClasses; ++c)
        {
            printf("%.4f ", host_array[b * numClasses + c]);
        }
        printf("\n");
    }

    free(host_array);
}

// Mnist Image Loader Function
// Converts byte format to the correct configuration then read image file content
unsigned char *LoadMNISTImages(const char *path, int *number_of_images)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror("Could not open image file");
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, f);
    // Fix and convert big endian to little for x86
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2051)
    {
        printf("Invalid magic number in image file: %d\n", magic_number);
        exit(1);
    }

    int numImages = 0, rows = 0, cols = 0;
    fread(&numImages, sizeof(int), 1, f);
    fread(&rows, sizeof(int), 1, f);
    fread(&cols, sizeof(int), 1, f);
    numImages = __builtin_bswap32(numImages);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    printf("Loading %d images of size %dx%d\n", numImages, rows, cols);

    unsigned char *images = (unsigned char *)malloc(numImages * rows * cols);
    fread(images, sizeof(unsigned char), numImages * rows * cols, f);
    fclose(f);

    *number_of_images = numImages;
    return images;
}

// Mnist Label Loader Function
// Converts byte format to the correct configuration then read label file content
unsigned char *LoadMnistLabels(const char *path, int *number_of_labels)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror("Could not open label file");
        exit(1);
    }
    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, f);
    // Fix and convert big endian to little for x86
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2049)
    {
        printf("Invalid magic number in label file: %d\n", magic_number);
        exit(1);
    }

    int numLabels = 0;
    fread(&numLabels, sizeof(int), 1, f);
    numLabels = __builtin_bswap32(numLabels);

    unsigned char *labels = (unsigned char *)malloc(numLabels);
    fread(labels, sizeof(unsigned char), numLabels, f);
    fclose(f);

    *number_of_labels = numLabels;
    return labels;
}

// Normalizes array value to 0.0 - 1.0
void Normalize(float *normalizedImg, unsigned char *images, int len)
{
    for (int i = 0; i < len; i++)
    {
        normalizedImg[i] = images[i] / 255.0f;
    }
}

// Encodes a class label to one-hot vector
void OneHotEncode(float *onehotLabels, unsigned char *labels, int len, int numClasses)
{
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < numClasses; j++)
        {
            onehotLabels[i * numClasses + j] = (labels[i] == j) ? 1.0f : 0.0f;
        }
    }
}

// ReLU function
float ReLU(float x)
{
    return x > 0 ? x : 0;
}

// Derivative of ReLU function
float D_ReLu(float x)
{
    return x >= 0 ? 1 : 0;
}

// Sigmoid Function
float Sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// TODO: Implement quick switching of activation fucntion
// Types to quickly switch between activation fucntions
typedef enum
{
    RELU,
    SIGMOID,
    NONE
} ActivationType;

// Struct for one layer
struct Layer
{
    int inFeature;
    int outFeature;
    float *Weight;
    float *Bias;
    float *X;
    float *z;
    ActivationType activation;
};

// Weight initialize function
// It sets the weight array to rand/max * sqroot(size) - (sqroot(size)/2)
void InitWeights(Layer *layer)
{
    int in = layer->inFeature;
    int out = layer->outFeature;
    int totalSize = in * out;

    float *h_weights = (float *)malloc(totalSize * sizeof(float));
    float scale = sqrtf(2.0f / totalSize);
    for (int i = 0; i < totalSize; i++)
    {
        h_weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
    CUDA_CHECK(cudaMalloc(&layer->Weight, totalSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(layer->Weight, h_weights, totalSize * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights);
}

// Bias initialize function
// Sets the bias array to zeros
void InitBias(struct Layer *layer)
{
    int out = layer->outFeature;

    float *h_bias = (float *)calloc(out, sizeof(float)); // initialized to zero already

    CUDA_CHECK(cudaMalloc(&layer->Bias, out * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(layer->Bias, h_bias, out * sizeof(float), cudaMemcpyHostToDevice));

    free(h_bias);
}

// Categorical Cross-Entropy is a loss function used for multi-class classification tasks.
// It measures the difference between the predicted probability distribution (from softmax) and the actual one-hot encoded label.
// The formula is: L = -sum(y_true[i] * log(y_pred[i])) for each class i
// - y_true[i] is 1 for the correct class, 0 otherwise (one-hot vector)
// - y_pred[i] is the predicted probability for class i
// This loss penalizes wrong confident predictions more heavily and encourages the model to assign high probability to the correct class.
float CalculateCCELoss(float *yPred, float *yTrue, int batchSize, int numClasses)
{
    float lossAvg = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float Loss = 0.0f;
        for (int j = 0; j < numClasses; j++)
        {
            int idx = i * numClasses + j;
            if (yTrue[idx] > 0.0f)
            {
                Loss += yTrue[idx] * logf(yPred[idx] + 1e-7f);
            }
        }
        Loss = -Loss;
        lossAvg += Loss;
    }
    // this affects the output gradient
    lossAvg = lossAvg / batchSize;
    return lossAvg;
}

// ==========CUDA Accelerated Functions and Kernels==========

// Kernel to intialize random array value
__global__ void __CU_init_random(float *data, int size, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

// Kernel to calculate the ReLU function
__global__ void __CU_ReLU(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// Kernel to calculate the derivative of the ReLU function
__global__ void __CU_relu_derivative(float *grad, float *x, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad[idx] *= (x[idx] > 0) ? 1.0f : 0.0f;
    }
}

// Kernel for Matrix Multiplication
// A: [M x K] - input matrix 1
// B: [K x N] - input matrix 2
// C: [M x N] - output matrix
__global__ void __CU_MatmulBatch(float *C, float *A, float *B, int M, int K, int N)
{
    // Declare row and col so the kernel knows where to target
    // can be read as row skip value + target col index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // safeguard so it doesnt exceed the thread count
    if (row < M && col < N)
    {
        // Matmul addition part
        float val = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            val += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

// Kernel for Matrix Transpose
// A: [M x K] - input matrix
// B: [K x M] - output matrix
__global__ void __CU_Transpose(float *B, const float *A, int M, int K)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // safeguard so it doesnt exceed the thread
    if (row < M && col < K)
    {
        B[col * M + row] = A[row * K + col];
    }
}

// Kernel fot Matrix Addition
// A: [M x N] - input matrix 1
// B: [M X N] - input matrix 2
// C: [M x N] - output matrix
__global__ void __CU_VecAdd(float *C, float *A, float *B, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // safeguard so it doesnt exceed the thread
    if (row < M && col < N)
    {
        C[row * N + col] = A[row * N + col] + B[col];
    }
}

// Unified Kernel for updating parameter
// This implementation uses Gradient Descent
__global__ void __CU_UpdateParameter(float *W, const float *dEdW, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        W[idx] -= learningRate * dEdW[idx];
    }
}

// Utility Kernel to compute squared sum
__global__ void __CU_ComputeSquaredSum(float *grad, float *partialSum, int size)
{
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (tid < size)
    {
        temp += grad[tid] * grad[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        partialSum[blockIdx.x] = cache[0];
}

// Utility Kernel to scale a value
__global__ void __CU_ReduceAndScale(float *partialSum, float *scale, int gridSize, float threshold)
{
    float normSq = 0.0f;
    for (int i = 0; i < gridSize; ++i)
    {
        normSq += partialSum[i];
    }

    float norm = sqrtf(normSq);
    if (norm > threshold)
    {
        *scale = threshold / norm;
    }
    else
    {
        *scale = 1.0f; // no scaling needed
    }
}

// Kernel to scale a gradient by some scalar value
__global__ void __CU_ScaleGradient(float *grad, int size, float *scale)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        grad[idx] *= *scale;
    }
}

// Kernel to clip gradient value and keep it in a certain maximum range
void __CU_ClipGradientValue(float *grad, int size, float threshold)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    float *d_partialSum, *d_scale;
    CUDA_CHECK(cudaMalloc(&d_partialSum, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale, sizeof(float)));

    // compute squared sum in blocks->reduce->scale
    __CU_ComputeSquaredSum<<<gridSize, blockSize>>>(grad, d_partialSum, size);
    __CU_ReduceAndScale<<<1, 1>>>(d_partialSum, d_scale, gridSize, threshold);
    __CU_ScaleGradient<<<gridSize, blockSize>>>(grad, size, d_scale);

    cudaFree(d_partialSum);
    cudaFree(d_scale);
}

// Kernel to compute the passing gradient on each layer(dEdz)
__global__ void __CU_Compute_dEdz(float *dEdz, const float *current_dEdy, const float *z, int size, bool apply_relu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float dydx = apply_relu ? (z[i] > 0.0f ? 1.0f : 0.0f) : 1.0f;
        dEdz[i] = current_dEdy[i] * dydx;
    }
}

// Kernel to compute the bias gradient value
__global__ void __CU_ComputeBiasGradient(float *dEdb, const float *dEdz, int batchSize, int outF)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < outF)
    {
        float sum = 0.0f;
        for (int b = 0; b < batchSize; b++)
        {
            sum += dEdz[b * outF + j];
        }
        dEdb[j] = sum;
    }
}

// Forward Propagation function
// logits: raw softmax predition
// layerSequence: struct of layers
// numLayers: the number of layers present in the struct
// input: truth data of the batch
// batchSize: size of the batch that is processed
void ForwardProp(float *logits, Layer *layerSequence, int numLayers, float *input, int batchSize)
{
    // Find max feature size for buffer allocation
    int maxFeature = 0;
    for (int i = 0; i < numLayers; i++)
    {
        if (layerSequence[i].inFeature > maxFeature)
            maxFeature = layerSequence[i].inFeature;
        if (layerSequence[i].outFeature > maxFeature)
            maxFeature = layerSequence[i].outFeature;
    }

    float *temp1;
    size_t bufferSize = batchSize * maxFeature * sizeof(float);
    CUDA_CHECK(cudaMalloc(&temp1, bufferSize));

    float *current = input;
    float *next = temp1;

    dim3 blockSize(16, 16);

    for (int layer = 0; layer < numLayers; layer++)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;
        float *W = layerSequence[layer].Weight;
        float *b = layerSequence[layer].Bias;

        dim3 gridSizeMatmul((outF + blockSize.x - 1) / blockSize.x,
                            (batchSize + blockSize.y - 1) / blockSize.y);
        __CU_MatmulBatch<<<gridSizeMatmul, blockSize>>>(next, current, W, batchSize, inF, outF);

        dim3 gridSizeVecAdd((outF + blockSize.x - 1) / blockSize.x,
                            (batchSize + blockSize.y - 1) / blockSize.y);
        __CU_VecAdd<<<gridSizeVecAdd, blockSize>>>(next, next, b, batchSize, outF);

        // Store results for backprop (allocate once outside if needed)
        if (!layerSequence[layer].z) // allocate once if not already
            CUDA_CHECK(cudaMalloc(&layerSequence[layer].z, batchSize * outF * sizeof(float)));
        if (!layerSequence[layer].X)
            CUDA_CHECK(cudaMalloc(&layerSequence[layer].X, batchSize * inF * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(layerSequence[layer].z, next, batchSize * outF * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(layerSequence[layer].X, current, batchSize * inF * sizeof(float), cudaMemcpyDeviceToDevice));

        if (layer < numLayers - 1)
        {
            int size = batchSize * outF;
            int threadsPerBlock = 256;
            int blocksReLU = (size + threadsPerBlock - 1) / threadsPerBlock;
            __CU_ReLU<<<blocksReLU, threadsPerBlock>>>(next, size);
        }

        // Swap buffers for next layer input
        float *tmp = current;
        current = next;
        next = tmp;
    }

    int outputSize = batchSize * layerSequence[numLayers - 1].outFeature;
    cudaMemcpy(logits, current, outputSize * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(temp1);
}

// Forward Propagation function
// layerSequence: struct of layers
// dEdy: The gradiet of the prediction and truth
// numLayers: the number of layers present in the struct
// batchSize: size of the batch that is processed
// learningRate: value of the update step
void BackwardProp(struct Layer *layerSequence, float *dEdy, int numLayers, int batchSize, float learningRate)
{
    // this is fine
    float *current_dEdy = dEdy;
    // CopyAndPrintDeviceArray(current_dEdy, batchSize, 10, "awiskiwiwi");
    for (int layer = numLayers - 1; layer >= 0; layer--)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;

        float *X = layerSequence[layer].X;
        //  z empty
        float *z = layerSequence[layer].z;
        // CopyAndPrintDeviceArray(z, batchSize, 10, "awiskiwiwi");
        float *W = layerSequence[layer].Weight;
        float *b = layerSequence[layer].Bias;

        // Allocate GPU buffers
        float *d_XT, *d_dEdz, *d_dEdW, *d_dEdb, *d_WT, *d_new_dEdy;

        CUDA_CHECK(cudaMalloc(&d_XT, batchSize * inF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dEdz, batchSize * outF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dEdW, inF * outF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dEdb, outF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WT, inF * outF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_new_dEdy, batchSize * inF * sizeof(float)));

        // Compute dEdz = dEdy * dReLU/dz
        int totalOut = batchSize * outF;
        int blockSize = 256;
        int gridSize = (totalOut + blockSize - 1) / blockSize;

        __CU_Compute_dEdz<<<gridSize, blockSize>>>(d_dEdz, current_dEdy, z, totalOut, (layer < numLayers - 1));

        // // X^T
        dim3 dimBlock(16, 16);
        dim3 dimGrid((inF + 15) / 16, (batchSize + 15) / 16);

        __CU_Transpose<<<dimGrid, dimBlock>>>(d_XT, X, batchSize, inF);
        // // dEdW = X^T @ dEdz
        __CU_MatmulBatch<<<dim3((outF + 15) / 16, (inF + 15) / 16), dim3(16, 16)>>>(d_dEdW, d_XT, d_dEdz, inF, batchSize, outF);
        // // Clip gradients
        __CU_ClipGradientValue(d_dEdW, inF * outF, 0.5f);
        // // dEdb = mean of dEdz over batch
        // CopyAndPrintDeviceArray(d_dEdW, batchSize, 10, ("awiskiwiwi"));
        __CU_ComputeBiasGradient<<<(outF + 255) / 256, 256>>>(d_dEdb, d_dEdz, batchSize, outF);
        // // Update W and b
        __CU_UpdateParameter<<<(inF * outF + 255) / 256, 256>>>(W, d_dEdW, learningRate, inF * outF);
        __CU_UpdateParameter<<<(outF + 255) / 256, 256>>>(b, d_dEdb, learningRate, outF);

        // // W^T
        __CU_Transpose<<<dim3((inF + 15) / 16, (outF + 15) / 16), dim3(16, 16)>>>(d_WT, W, inF, outF);
        // CopyAndPrintDeviceArray(d_WT, 1, 10, ("awiskiwiwi"));

        // // new_dEdy = dEdz @ W^T
        __CU_MatmulBatch<<<dim3((inF + 15) / 16, (batchSize + 15) / 16), dim3(16, 16)>>>(d_new_dEdy, d_dEdz, d_WT, batchSize, outF, inF);

        // there is a zero somewhere here
        if (layer != numLayers - 1)
        {
            cudaFree(current_dEdy);
        }

        current_dEdy = d_new_dEdy;

        // // Free temporary GPU buffers
        cudaFree(d_XT);
        cudaFree(d_dEdz);
        cudaFree(d_dEdW);
        cudaFree(d_dEdb);
        cudaFree(d_WT);
        // // do NOT free d_new_dEdy here
    }
    cudaFree(current_dEdy); // free after last layer
}

// Softmax Kernel implementation of the softmax Function
// raw values->probability distribution
__global__ void __CU_Softmax(float *input, float *output, int batch_size, int size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size)
    {
        int offset = b * size;
        float max_val = input[offset];
        for (int i = 1; i < size; ++i)
        {
            max_val = fmaxf(max_val, input[offset + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i)
        {
            output[offset + i] = expf(input[offset + i] - max_val);
            sum += output[offset + i];
        }

        for (int i = 0; i < size; ++i)
        {
            output[offset + i] = fmaxf(output[offset + i] / sum, 1e-7f); // clamp to avoid log(0)
        }
    }
}

void CU_SoftMaxBatch(float *device_input, float *device_output, int batchSize, int numClasses)
{
    int threads = 128;
    int blocks = (batchSize + threads - 1) / threads;

    __CU_Softmax<<<blocks, threads>>>(device_input, device_output, batchSize, numClasses);
    // CUDA_CHECK(cudaGetLastError()); // cillegal memeroy access here
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel to calcualte the gradient of the crossentropy loss fucntion
__global__ void __CU_CrossEntropyGradient(float *dEdy, const float *logits, const float *y_batch, int total, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        dEdy[idx] = (logits[idx] - y_batch[idx]) / batchSize;
    }
}

// Kernel wrapper function to calculate the crossentropy gradient
void CU_CrossEntropyGradient(float *dEdy, const float *d_logits, const float *d_y_batch, int batchSize, int numClasses)
{
    int total = batchSize * numClasses;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    __CU_CrossEntropyGradient<<<gridSize, blockSize>>>(dEdy, d_logits, d_y_batch, total, batchSize);
}

// Function to predict a label given an mnist(28x28) image
void PredictSingle(int inputIndex, struct Layer *layerSequence, int numLayers, float *X, unsigned char *label, int numClasses)
{
    int inputSize = 784;

    float *deviceInput = &X[inputIndex * inputSize];
    float *logitsSingle = (float *)calloc(numClasses, sizeof(float));
    float *singleInput = (float *)calloc(inputSize, sizeof(float));
    float *logitsRaw;
    float *logitsSoftmax;
    CUDA_CHECK(cudaMalloc(&logitsRaw, sizeof(float) * numClasses));
    CUDA_CHECK(cudaMalloc(&logitsSoftmax, sizeof(float) * numClasses));
    ForwardProp(logitsRaw, layerSequence, numLayers, deviceInput, 1);
    CU_SoftMaxBatch(logitsRaw, logitsSoftmax, 1, layerSequence[numLayers - 1].outFeature);
    CUDA_CHECK(cudaMemcpy(logitsSingle, logitsSoftmax, sizeof(float) * numClasses, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(singleInput, deviceInput, sizeof(float) * inputSize, cudaMemcpyDeviceToHost));

    printf("=======================\n");
    printf("Predicted probabilities:\n");
    int predictedClass = 0;
    float maxProb = logitsSingle[0];
    for (int i = 1; i < numClasses; i++)
    {
        if (logitsSingle[i] > maxProb)
        {
            maxProb = logitsSingle[i];
            predictedClass = i;
        }
    }

    printf("Predicted class: %d\n", predictedClass);
    printf("True label: %d\n", label[inputIndex]);
    for (int i = 0; i < 28 * 28; i++)
    {
        if (singleInput[i] > 0)
        {
            if (singleInput[i] > 0.5)
            {
                printf("# ");
            }
            else
            {
                printf("- ");
            }
        }
        else
        {
            printf("  ");
        }
        if ((i + 1) % 28 == 0)
            printf("\n");
    }
    free(logitsSingle);
    free(singleInput);
    CUDA_CHECK(cudaFree(logitsRaw));
    CUDA_CHECK(cudaFree(logitsSoftmax));
}

// Main program
int main(int argc, char const *argv[])
{
    srand(time(NULL));

    // Data Source: https://github.com/cvdfoundation/mnist?tab=readme-ov-file
    const char *imagePath = "/app/dataset/MNIST/train-images-idx3-ubyte";
    const char *labelPath = "/app/dataset/MNIST/train-labels-idx1-ubyte";

    int numImages = 0;
    int numLabels = 0;
    unsigned char *images = LoadMNISTImages(imagePath, &numImages);
    unsigned char *labels = LoadMnistLabels(labelPath, &numLabels);

    if (numImages != numLabels)
    {
        printf("Mismatch between images and labels count.\n");
        exit(1);
    }

    float *normalizedImg = (float *)calloc(28 * 28 * numImages, sizeof(float));
    float *onehotLabels = (float *)calloc(10 * numLabels, sizeof(float));
    // Normalize images to a 0-1 range
    Normalize(normalizedImg, images, 28 * 28 * numImages);
    // Encode labels to one ot vecotrs
    OneHotEncode(onehotLabels, labels, numLabels, 10);

    float *X, *y;
    CUDA_CHECK(cudaMalloc(&X, sizeof(float) * 28 * 28 * numImages));
    CUDA_CHECK(cudaMemcpy(X, normalizedImg, sizeof(float) * 28 * 28 * numImages, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&y, sizeof(float) * 10 * numLabels));
    CUDA_CHECK(cudaMemcpy(y, onehotLabels, sizeof(float) * 10 * numLabels, cudaMemcpyHostToDevice));
    // Print Example Image
    printf("Example image\n");
    printf("Label[0] = %d\n", labels[0]);
    for (int i = 0; i < 28 * 28; i++)
    {
        if (normalizedImg[i] > 0)
        {
            printf("# ");
        }
        else
        {
            printf("- ");
        }
        if ((i + 1) % 28 == 0)
            printf("\n");
    }

    // Build Layers
    struct Layer layerSequence[] = {
        {.inFeature = 784, .outFeature = 4096, .activation = RELU},
        {.inFeature = 4096, .outFeature = 10, .activation = NONE}};
    int numLayers = sizeof(layerSequence) / sizeof(layerSequence[0]);

    printf("Log: There are %d layers in this network\n", numLayers);
    // Intialize weight and bias values
    for (int i = 0; i < numLayers; i++)
    {
        InitWeights(&layerSequence[i]);
        InitBias(&layerSequence[i]);
    }
    int numClasses = 10;
    int totalSamples = 10000;
    int epoch_count = 20;
    int batchSize = 32;
    int batchCount = totalSamples / batchSize;

    // Allocate buffers for logits and output gradient
    float *logits;
    float *logits_softmax;
    float *dEdy;
    CUDA_CHECK(cudaMalloc(&logits, sizeof(float) * batchSize * numClasses));
    CUDA_CHECK(cudaMalloc(&logits_softmax, sizeof(float) * batchSize * numClasses));
    CUDA_CHECK(cudaMalloc(&dEdy, sizeof(float) * batchSize * numClasses));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    float *logits_host = (float *)malloc(batchSize * numClasses * sizeof(float));
    float *y_batch_host = (float *)malloc(batchSize * numClasses * sizeof(float));
    for (int epoch = 0; epoch < epoch_count; epoch++)
    {
        double totalLoss = 0.0;

        for (int b = 0; b < batchCount; b++)
        {
            float *X_batch = &X[b * batchSize * 784];
            float *y_batch = &y[b * batchSize * numClasses];
            ForwardProp(logits, layerSequence, numLayers, X_batch, batchSize);
            CU_SoftMaxBatch(logits, logits_softmax, batchSize, numClasses);
            CU_CrossEntropyGradient(dEdy, logits_softmax, y_batch, batchSize, numClasses);
            CUDA_CHECK(cudaMemcpy(logits_host, logits_softmax, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y_batch_host, y_batch, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost));
            totalLoss += CalculateCCELoss(logits_host, y_batch_host, batchSize, numClasses);
            BackwardProp(layerSequence, dEdy, numLayers, batchSize, 0.05f);
        }

        printf("Epoch #%d, Avg Loss: %.6f\n", epoch + 1, totalLoss / batchCount);
    }
    free(logits_host);
    free(y_batch_host);

    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate duration in seconds with milliseconds
    double training_time = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\nTotal training time: %.2f sec\n", training_time);

    // Print predictions and check learning result
    PredictSingle(1250, layerSequence, numLayers, X, labels, numClasses);
    PredictSingle(49230, layerSequence, numLayers, X, labels, numClasses);
    PredictSingle(25122, layerSequence, numLayers, X, labels, numClasses);

    // Free memory - this throws illegal for some reason????
    CUDA_CHECK(cudaFree(logits));
    CUDA_CHECK(cudaFree(logits_softmax));
    CUDA_CHECK(cudaFree(dEdy));

    free(images);
    free(labels);
    free(normalizedImg);
    free(onehotLabels);

    printf("=====program end=====\n");
    return 0;
}