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

// Global Variable to set GPU Acceleration
// Initialized as False, do not change this
// Change the one on the main fcuntion instead
bool gpuInference = false;

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

// Softmax Function
// modified to accept batches
void SoftMaxBatch(float *x, int batchSize, int size)
{
    for (int b = 0; b < batchSize; ++b)
    {
        float *row = &x[b * size];

        // Step 1: find max for numerical stability
        float max_val = row[0];
        for (int i = 1; i < size; i++)
        {
            if (row[i] > max_val)
                max_val = row[i];
        }

        // Step 2: exponentiate and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }

        // Step 3: normalize
        for (int i = 0; i < size; i++)
        {
            row[i] /= sum;
        }
    }
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

// Returns a random floating point(decimal) value
float randFloat()
{
    return ((float)rand()) / RAND_MAX;
}

// A: [M x K] - input matrix 1
// B: [K x N] - input matrix 2
// C: [M x N] - output
void MatmulBatch(float *C, float *A, float *B, int M, int K, int N)
{
    for (int m = 0; m < M; ++m)
    {
        float *a = &A[m * K];
        float *c = &C[m * N];

        for (int n = 0; n < N; ++n)
        {
            float val = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                val += B[k * N + n] * a[k];
            }
            c[n] = val;
        }
    }
}

// A: [M x K] - input matrix
// B: [K x M] - output matrix
void Transpose(float *B, const float *A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // B[j][i] = A[i][j]
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void InitWeights(struct Layer *layer)
{
    int in = layer->inFeature;
    int out = layer->outFeature;

    // Allocate a 1D array of size (inFeature * outFeature)
    // Weight is shaped (inFeature, outFeature)
    layer->Weight = (float *)malloc(in * out * sizeof(float));
    for (int i = 0; i < out; ++i)
    {
        for (int j = 0; j < in; ++j)
        {
            layer->Weight[i * in + j] = randFloat() * sqrtf(2.0f / (in + out)); // or 0.0
        }
    }
}

void InitBias(struct Layer *layer)
{
    int out = layer->outFeature;

    // Allocate a 1D array of size outFeature
    // Bias is shaped (outFeature)
    layer->Bias = (float *)calloc(out, sizeof(float));

    for (int i = 0; i < out; ++i)
    {
        layer->Bias[i] = 0;
    }
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

// Cuda Matmul wrapper function
void CU_MatmulBatch(float *C, float *A, float *B, int M, int K, int N)
{
    // Allocate Memory
    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    // Copy from CPU->GPU
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Declare block and grid dimensions(parameters)
    // Hierarchy: grid->block->thread
    // remember: cuda grid is written as (col, row) so its x first
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Call cuda Kernel
    __CU_MatmulBatch<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, K, N);

    // Wait Until all execution is finished
    cudaDeviceSynchronize();

    // Copy from GPU->CPU
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

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

// Cuda Transpose wrapper function
void CU_Transpose(float *B, const float *A, int M, int K)
{
    float *d_A;
    // B is out
    float *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, M * K * sizeof(float));
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16);
    // remember: cuda grid is col, row, so its x first
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    __CU_Transpose<<<gridDim, blockDim>>>(d_B, d_A, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

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

void CU_VecAdd(float *C, float *A, float *B, int M, int N)
{
    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    // grid->block->thread
    // max
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    __CU_VecAdd<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void __CU_UpdateParameter(float *W, const float *dEdW, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        W[idx] -= learningRate * dEdW[idx];
    }
}

void CU_UpdateParameter(float *W, const float *dEdW, float learningRate, int M, int K)
{
    float *d_W, *d_dEdW;
    int size = M * K;
    cudaMalloc(&d_W, size * sizeof(float));
    cudaMalloc(&d_dEdW, size * sizeof(float));

    cudaMemcpy(d_W, W, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dEdW, dEdW, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    // just one grid
    int gridSize = (size + blockSize - 1) / blockSize;

    __CU_UpdateParameter<<<gridSize, blockSize>>>(d_W, d_dEdW, learningRate, size);
    cudaDeviceSynchronize();

    cudaMemcpy(W, d_W, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_W);
    cudaFree(d_dEdW);
}

void ForwardProp(float *logits, struct Layer *layerSequence, int numLayers, float *input, int batchSize)
{

    // allocate 2 buffers
    float *temp1 = (float *)calloc(784 * batchSize, sizeof(float));
    float *temp2 = (float *)calloc(784 * batchSize, sizeof(float));
    float *current = input;
    float *next = temp1;
    for (int layer = 0; layer < numLayers; layer++)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;
        float *W = layerSequence[layer].Weight;
        float *b = layerSequence[layer].Bias;

        // 32->the neuron count
        // 30->the connection on each neuron, same as length of a single training item

        // Multiply current @ W  -> next
        if (gpuInference)
        {
            CU_MatmulBatch(next, current, W, batchSize, inF, outF);
            CU_VecAdd(next, next, b, batchSize, outF);
        }
        else
        {
            MatmulBatch(next, current, W, batchSize, inF, outF);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outF; j++)
                {
                    next[i * outF + j] += b[j];
                }
            }
        }

        layerSequence[layer].z = (float *)calloc(batchSize * outF, sizeof(float));
        layerSequence[layer].X = (float *)calloc(batchSize * inF, sizeof(float));
        memcpy(layerSequence[layer].z, next, batchSize * outF * sizeof(float));
        memcpy(layerSequence[layer].X, current, batchSize * inF * sizeof(float));

        // Apply ReLU element-wise
        if (layer < numLayers - 1)
        {
            for (int i = 0; i < batchSize * outF; i++)
            {
                next[i] = ReLU(next[i]);
            }
        }

        float *temp = current;
        current = next;
        if (layer < numLayers - 1)
        {
            next = (temp == temp1) ? temp2 : temp1;
        }
    }

    // plain copy value
    memcpy(logits, next, batchSize * layerSequence[numLayers - 1].outFeature * sizeof(float));
    free(temp1);
    free(temp2);
}

void ClipGradientValue(float *grad, int size, float threshold)
{
    float norm = 0.0f;
    for (int i = 0; i < size; i++)
    {
        norm += grad[i] * grad[i];
    }
    norm = sqrtf(norm);

    if (norm > threshold)
    {
        float scale = threshold / norm;
        for (int i = 0; i < size; i++)
        {
            grad[i] *= scale;
        }
    }
}

void BackwardProp(struct Layer *layerSequence, float *dEdy, int numLayers, int batchSize, float learningRate)
{
    float *current_dEdy = dEdy;

    for (int layer = numLayers - 1; layer >= 0; layer--)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;
        float *W = layerSequence[layer].Weight;
        float *b = layerSequence[layer].Bias;
        float *X = layerSequence[layer].X;
        float *z = layerSequence[layer].z;
        float *dydx = (float *)calloc(batchSize * outF, sizeof(float));
        float *dEdx = (float *)calloc(batchSize * outF, sizeof(float));
        float *dEdW = (float *)calloc(inF * outF, sizeof(float));
        float *dEdb = (float *)calloc(outF, sizeof(float));
        // X is [inF x batchSize], XT is [batchSize x inF]
        float *XT = (float *)malloc(inF * batchSize * sizeof(float));
        // W is [inF x outF], WT is [outF x inF]
        float *WT = (float *)malloc(outF * inF * sizeof(float));
        float *new_dEdy = (float *)malloc(batchSize * inF * sizeof(float));
        // Clip value/the maximum of a gradient value can reach
        float clipValue = 0.5;

        for (int i = 0; i < batchSize * outF; i++)
        {
            dydx[i] = (layer < numLayers - 1) ? D_ReLu(z[i]) : 1.0f;
            dEdx[i] = current_dEdy[i] * dydx[i];
        }

        if (gpuInference)
        {
            CU_Transpose(XT, X, batchSize, inF);
            // Weight update value
            // dEdx @ self.W.T
            CU_MatmulBatch(dEdW, XT, dEdx, inF, batchSize, outF);
            ClipGradientValue(dEdW, inF * outF, clipValue);
            // Bias update value
            for (int j = 0; j < outF; j++)
            {
                dEdb[j] = 0.0f;
                for (int b = 0; b < batchSize; b++)
                {
                    dEdb[j] += dEdx[b * outF + j];
                }
            }
            CU_UpdateParameter(W, dEdW, learningRate, inF, outF);
            CU_UpdateParameter(b, dEdb, learningRate, outF, 1);
            CU_Transpose(WT, W, inF, outF);
            CU_MatmulBatch(new_dEdy, dEdx, WT, batchSize, outF, inF);
        }
        else
        {
            Transpose(XT, X, batchSize, inF);
            // dEdx @ self.W.T
            MatmulBatch(dEdW, XT, dEdx, inF, batchSize, outF);
            ClipGradientValue(dEdW, inF * outF, clipValue);
            // printf("layer %d: %.2lf, %.2lf, %.2lf\n", layer, X[0], W[0], z[0]);
            // printf("layer %d deriv: %.2lf, %.2lf, %.2lf\n", layer, dEdx[0], dEdW[0], current_dEdy[0]);
            for (int j = 0; j < outF; j++)
            {
                dEdb[j] = 0.0f;
                for (int b = 0; b < batchSize; b++)
                {
                    dEdb[j] += dEdx[b * outF + j];
                }
            }
            for (int i = 0; i < inF * outF; i++)
            {
                W[i] -= learningRate * dEdW[i];
            }
            for (int j = 0; j < outF; j++)
            {
                b[j] -= learningRate * dEdb[j];
            }
            Transpose(WT, W, inF, outF);
            MatmulBatch(new_dEdy, dEdx, WT, batchSize, outF, inF);
        }

        if (layer != numLayers - 1)
        {
            free(current_dEdy);
        }

        current_dEdy = new_dEdy;

        free(XT);
        free(WT);
        free(dydx);
        free(dEdx);
        free(dEdW);
        free(dEdb);
    }

    free(current_dEdy);
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

void PredictSingle(int inputIndex, struct Layer *layerSequence, int numLayers, float *X, unsigned char *label, int numClasses)
{
    int inputSize = 784;

    float *singleInput = &X[inputIndex * inputSize];
    float *logitsSingle = (float *)calloc(numClasses, sizeof(float));

    ForwardProp(logitsSingle, layerSequence, numLayers, singleInput, 1);
    SoftMaxBatch(logitsSingle, 1, layerSequence[numLayers - 1].outFeature);
    printf("=======================\n");
    printf("Predicted probabilities:\n");
    // printf("Porbabilities");
    // for (int i = 0; i < numClasses; i++)
    // {
    //     printf("x%.4f ", i, logitsSingle[i]);
    // }
    // printf("\n");

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
            printf("# ");
        }
        else
        {
            printf("- ");
        }
        if ((i + 1) % 28 == 0)
            printf("\n");
    }
    free(logitsSingle);
}

void Train(struct Layer layer, int epoch)
{
    for (int i = 0; i < epoch; i++)
    {
        printf("Epoch %d, Loss: nan, Acc: nan\n", epoch);
    }
}

int main(int argc, char const *argv[])
{
    // Toggle this for GPU Acceleration
    gpuInference = true;

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

    float *X = normalizedImg;
    float *y = onehotLabels;

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
    srand(time(NULL));

    // Build Layers
    struct Layer layerSequence[] = {
        {.inFeature = 784, .outFeature = 64, .activation = RELU},
        {.inFeature = 64, .outFeature = 10, .activation = NONE}};
    int numLayers = sizeof(layerSequence) / sizeof(layerSequence[0]);

    printf("Log: There are %d layers in this network\n", numLayers);

    // Intialize weight and bias values
    for (int i = 0; i < numLayers; i++)
    {
        InitWeights(&layerSequence[i]);
        InitBias(&layerSequence[i]);
    }

    float loss = 0;
    int numClasses = 10;
    int batchSize = 48000;
    // Initialize buffer
    float *logits = (float *)calloc(numClasses * batchSize, sizeof(float));
    float *dEdy = (float *)calloc(batchSize * numClasses, sizeof(float));

    clock_t t;
    t = clock();

    int epoch_count = 100;
    for (int i = 0; i < epoch_count; i++)
    {
        // Calculate initial prediction
        ForwardProp(logits, layerSequence, numLayers, X, batchSize);
        // Use Softmax to get probabilities of each class
        SoftMaxBatch(logits, batchSize, layerSequence[numLayers - 1].outFeature);
        // Calculate Output gradient
        // Since this is full batch gradient descent, divide by the batchSize
        for (int j = 0; j < batchSize * numClasses; j++)
        {
            dEdy[j] = (logits[j] - y[j]) / batchSize;
        }
        // Calculate Cross Entropy Loss
        loss = CalculateCCELoss(logits, y, batchSize, numClasses);
        printf("epoch #%d, loss: %lf\n", i + 1, loss);
        // Using Output Gradient and backpropagation, update parameters
        BackwardProp(layerSequence, dEdy, numLayers, batchSize, 0.1);
    }

    t = clock() - t;

    // Print predictions and check learning result
    PredictSingle(1250, layerSequence, numLayers, X, labels, numClasses);
    PredictSingle(4700, layerSequence, numLayers, X, labels, numClasses);
    PredictSingle(2512, layerSequence, numLayers, X, labels, numClasses);

    // Free memory
    free(logits);
    free(dEdy);
    free(images);
    free(labels);
    free(normalizedImg);
    free(onehotLabels);

    // Calculate time taken by the training
    double time_taken = ((double)t) / CLOCKS_PER_SEC;
    printf("log: it took %.2lf seconds to execute %d epochs\n", time_taken, epoch_count);
    printf("=====program end=====\n");
    return 0;
}
