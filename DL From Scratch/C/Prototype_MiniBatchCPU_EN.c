// ============ MNIST Classifier from Scratch using Plain CPU ============
// This file demonstrates the use of GPU acceleration for training a neural network.
// It applies a simple classifier to the MNIST dataset, leveraging standard library to perform
// key operations such as matrix multiplication, vector addition, and matrix transpotition efficiently on the GPU.
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

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

// Softmax Function
// modified to accept batches
void SoftMaxBatch(float *x, int batchSize, int size)
{
    for (int b = 0; b < batchSize; ++b)
    {
        float *row = &x[b * size];

        float max_val = row[0];
        for (int i = 1; i < size; i++)
        {
            if (row[i] > max_val)
                max_val = row[i];
        }

        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }

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
    int totalSize = in * out;
    float scale = sqrtf(2.0f / totalSize);
    for (int i = 0; i < totalSize; i++)
    {
        layer->Weight[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
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

void ForwardProp(float *logits, struct Layer *layerSequence, int numLayers, float *input, int batchSize)
{

    // allocate 2 buffers
    float *temp1 = (float *)calloc(4096 * batchSize, sizeof(float));
    float *temp2 = (float *)calloc(4096 * batchSize, sizeof(float));
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

        MatmulBatch(next, current, W, batchSize, inF, outF);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outF; j++)
            {
                next[i * outF + j] += b[j];
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

void BackwardProp(struct Layer *layerSequence, float *dEdy, int numLayers, int batchSize, float learningRate)
{
    float *current_dEdy = dEdy;
    for (int layer = numLayers - 1; layer >= 0; layer--)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;
        float *W = layerSequence[layer].Weight;
        float *X = layerSequence[layer].X;
        float *z = layerSequence[layer].z;
        float *dydz = calloc(batchSize * outF, sizeof(float));
        float *dEdz = calloc(batchSize * outF, sizeof(float));
        float *dEdW = calloc(inF * outF, sizeof(float));
        float *dEdb = calloc(outF, sizeof(float));

        for (int i = 0; i < batchSize * outF; i++)
        {
            dydz[i] = (layer < numLayers - 1) ? D_ReLu(z[i]) : 1.0f;
            dEdz[i] = current_dEdy[i] * dydz[i];
        }
        float *XT = malloc(inF * batchSize * sizeof(float));
        Transpose(XT, X, batchSize, inF);
        MatmulBatch(dEdW, XT, dEdz, inF, batchSize, outF);
        free(XT);
        // gradient clip = 5
        // ClipGradientValue(dEdW, inF * outF, 0.5f);

        // Weight update
        for (int i = 0; i < inF * outF; i++)
        {
            W[i] -= learningRate * dEdW[i];
        }

        for (int j = 0; j < outF; j++)
        {
            dEdb[j] = 0.0f;
            for (int b = 0; b < batchSize; b++)
            {
                dEdb[j] += dEdz[b * outF + j];
            }
        }

        //  Bias update
        for (int j = 0; j < outF; j++)
        {
            layerSequence[layer].Bias[j] -= learningRate * dEdb[j];
        }

        // dEdz @ self.W.T
        float *WT = malloc(outF * inF * sizeof(float));
        Transpose(WT, W, inF, outF);
        // W is [inF x outF], WT is [outF x inF]
        float *new_dEdy = malloc(batchSize * inF * sizeof(float));
        MatmulBatch(new_dEdy, dEdz, WT, batchSize, outF, inF);
        free(WT);

        if (layer != numLayers - 1)
        {
            free(current_dEdy);
        }

        current_dEdy = new_dEdy;

        free(dydz);
        free(dEdz);
        free(dEdW);
        free(dEdb);
    }

    free(current_dEdy);
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

int main(int argc, char const *argv[])
{

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
        {.inFeature = 784, .outFeature = 32, .activation = RELU},
        {.inFeature = 32, .outFeature = 10, .activation = NONE}};
    int numLayers = sizeof(layerSequence) / sizeof(layerSequence[0]);

    printf("Log: There are %d layers in this network\n", numLayers);

    // Intialize weight and bias values
    for (int i = 0; i < numLayers; i++)
    {
        InitWeights(&layerSequence[i]);
        InitBias(&layerSequence[i]);
    }

    // Define HyperParameters
    int numClasses = 10;
    int totalSamples = 10000;
    int epoch_count = 20;
    int batchSize = 32;
    int batchCount = totalSamples / batchSize;

    // Allocate buffers for logits and output gradient
    float *logits = (float *)calloc(batchSize * numClasses, sizeof(float));
    float *dEdy = (float *)calloc(batchSize * numClasses, sizeof(float));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Do minibatch gradient descent
    for (int epoch = 0; epoch < epoch_count; epoch++)
    {
        double totalLoss = 0.0;

        for (int b = 0; b < batchCount; b++)
        {
            float *X_batch = &X[b * batchSize * 784];
            float *y_batch = &y[b * batchSize * numClasses];
            ForwardProp(logits, layerSequence, numLayers, X_batch, batchSize);
            SoftMaxBatch(logits, batchSize, numClasses);
            int total = batchSize * numClasses;
            memset(dEdy, 0, sizeof(float) * total);
            for (int i = 0; i < total; i++)
            {
                dEdy[i] = (logits[i] - y_batch[i]) / batchSize;
            }
            totalLoss += CalculateCCELoss(logits, y_batch, batchSize, numClasses);
            BackwardProp(layerSequence, dEdy, numLayers, batchSize, 0.05f);
        }

        printf("Epoch #%d, Avg Loss: %.6f\n", epoch + 1, totalLoss / batchCount);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate duration in seconds with milliseconds
    double training_time = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;

    // Print predictions and check learning result
    PredictSingle(1250, layerSequence, numLayers, X, labels, numClasses);
    PredictSingle(49000, layerSequence, numLayers, X, labels, numClasses);
    PredictSingle(2512, layerSequence, numLayers, X, labels, numClasses);

    // Free memory
    free(logits);
    free(dEdy);
    free(images);
    free(labels);
    free(normalizedImg);
    free(onehotLabels);

    printf("\nTotal training time: %.2f sec\n", training_time);
    printf("=====program end=====\n");
    return 0;
}
