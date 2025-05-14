#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

// https://github.com/cvdfoundation/mnist?tab=readme-ov-file

unsigned char *load_mnist_images(const char *path, int *number_of_images)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror("Could not open image file");
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, f);
    // Fix, convert big endian to little for x86
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2051)
    {
        printf("Invalid magic number in image file: %d\n", magic_number);
        exit(1);
    }

    int num_images = 0, rows = 0, cols = 0;
    fread(&num_images, sizeof(int), 1, f);
    fread(&rows, sizeof(int), 1, f);
    fread(&cols, sizeof(int), 1, f);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    printf("Loading %d images of size %dx%d\n", num_images, rows, cols);

    unsigned char *images = malloc(num_images * rows * cols);
    fread(images, sizeof(unsigned char), num_images * rows * cols, f);
    fclose(f);

    *number_of_images = num_images;
    return images;
}

unsigned char *load_mnist_labels(const char *path, int *number_of_labels)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror("Could not open label file");
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, f);
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2049)
    {
        printf("Invalid magic number in label file: %d\n", magic_number);
        exit(1);
    }

    int numLabels = 0;
    fread(&numLabels, sizeof(int), 1, f);
    numLabels = __builtin_bswap32(numLabels);

    unsigned char *labels = malloc(numLabels);
    fread(labels, sizeof(unsigned char), numLabels, f);
    fclose(f);

    *number_of_labels = numLabels;
    return labels;
}

float relu(float x)
{
    return x > 0 ? x : 0;
}

float relu_derivative(float x)
{
    return x >= 0 ? 1 : 0;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void softmax_batch(float *x, int batchSize, int size)
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

typedef enum
{
    RELU,
    SIGMOID,
    NONE
} ActivationType;

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

float randFloat()
{
    return ((float)rand()) / RAND_MAX; // returns 0.0 to 1.0
}

// A: [M x K] - batch of inputs
// B: [K x N] - weights
// C: [M x N] - output
void matmul_batch(float *C, float *A, float *B, int M, int K, int N)
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

// size here is the size of a and b itself
void DotProduct(float *out, float *a, float *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = a[i] * b[i];
    }
}

void transpose(float *out, const float *in, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // out[j][i] = in[i][j]
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

void initWeights(struct Layer *layer)
{
    int in = layer->inFeature;
    int out = layer->outFeature;

    // Allocate a 1D array of size (inFeature * outFeature)
    layer->Weight = malloc(in * out * sizeof(float));
    // printf("\n\n=======weight=========\n\n");

    for (int i = 0; i < out; ++i)
    {
        for (int j = 0; j < in; ++j)
        {
            // Row-major: W[i * in + j] => weight from input j to output i
            layer->Weight[i * in + j] = randFloat() * sqrtf(2.0f / (in + out)); // or 0.0
            // printf(" <%d> ", i * in + j);
            // printf("%.2lf ", layer->Weight[i * in + j]);
        }
        // printf("\n\n");
    }
}

float CalculateCCELoss(float *yPred, float *yTrue, int batchSize, int numClasses)
{
    float lossAvg = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float Loss = 0.0f;
        for (int j = 0; j < numClasses; j++)
        {
            int idx = i * numClasses + j;
            // Only compute log where yTrue is 1 (for one-hot)
            if (yTrue[idx] > 0.0f)
            {
                // added epsilon for stability
                Loss += yTrue[idx] * logf(yPred[idx] + 1e-7f);
            }
        }
        Loss = -Loss;
        lossAvg += Loss;
        // printf("Sample %d Loss: %.4f\n", i, Loss);
    }
    lossAvg = lossAvg / batchSize;
    return lossAvg;
    // printf("loss: %lf\n", lossAvg);
}

void ForwardProp(float *logits, struct Layer *layerSequence, int numLayers, float *input, int batchSize)
{
    // printf("\n\n=================input===========\n\n");
    // for (int i = 0; i < 1; ++i)
    // {
    //     for (int j = 0; j < layerSequence[0].inFeature; ++j)
    //     {
    //         printf("%.2f ", input[i * layerSequence[0].inFeature + j]);
    //     }
    //     printf("\n\n");
    // }
    // FIX maybe
    float *temp1 = calloc(784 * batchSize, sizeof(float)); // scratch buffer 1
    float *temp2 = calloc(784 * batchSize, sizeof(float)); // scratch buffer 2
    float *current = input;
    float *next = temp1;
    for (int layer = 0; layer < numLayers; layer++)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;
        float *W = layerSequence[layer].Weight;

        // 32->the neuron count
        // 30->the connection on each neuron, same as length of a single training item
        // printf("Layer %d (%d → %d):\n", layer, inF, outF);

        // Multiply W * current -> next
        matmul_batch(next, current, W, batchSize, inF, outF);
        // store next in z, idk if thisll break or not, this'll probably bite me back in the future :D
        // TODO: make this safe
        layerSequence[layer].z = calloc(batchSize * outF, sizeof(float));
        layerSequence[layer].X = calloc(batchSize * inF, sizeof(float));
        memcpy(layerSequence[layer].z, next, batchSize * outF * sizeof(float));
        memcpy(layerSequence[layer].X, current, batchSize * inF * sizeof(float));
        if (layer < numLayers - 1)
        {
            for (int i = 0; i < batchSize * outF; ++i)
            {
                next[i] = relu(next[i]); // Apply ReLU element-wise
            }
        }

        float *temp = current;
        current = next;
        if (layer < numLayers - 1)
        {
            next = (temp == temp1) ? temp2 : temp1;
        }
    }
    softmax_batch(next, batchSize, layerSequence[numLayers - 1].outFeature);

    // printf("\n\nafter softmax\n\n");
    // for (int i = 0; i < 1; i++)
    // {
    //     /* code */
    //     for (int j = 0; j < layerSequence[1].outFeature; j++)
    //     {
    //         printf("%.2lf ", next[i * layerSequence[1].outFeature + j]);
    //     }
    //     printf("\n\n");
    // }

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

        float *dydx = calloc(batchSize * outF, sizeof(float));
        float *dEdx = calloc(batchSize * outF, sizeof(float));
        float *dEdW = calloc(inF * outF, sizeof(float));
        float *dEdb = calloc(outF, sizeof(float));

        for (int i = 0; i < batchSize * outF; i++)
        {
            dydx[i] = (layer < numLayers - 1) ? relu_derivative(z[i]) : 1.0f;
            dEdx[i] = current_dEdy[i] * dydx[i];
        }

        float *XT = malloc(inF * batchSize * sizeof(float));
        transpose(XT, X, batchSize, inF);
        matmul_batch(dEdW, XT, dEdx, inF, batchSize, outF);
        free(XT);

        // Gradient descent weight update
        for (int i = 0; i < inF * outF; i++)
        {
            W[i] -= learningRate * dEdW[i];
        }

        //  Bias update
        // for (int j = 0; j < outF; j++)
        // {
        //     layerSequence[layer].Bias[j] -= learningRate * dEdb[j] / batchSize;
        // }

        // dEdx @ self.W.T
        float *WT = malloc(outF * inF * sizeof(float));
        transpose(WT, W, inF, outF);
        // W is [inF x outF], WT is [outF x inF]
        float *new_dEdy = malloc(batchSize * inF * sizeof(float));
        matmul_batch(new_dEdy, dEdx, WT, batchSize, outF, inF);
        free(WT);

        if (layer != numLayers - 1)
        {
            free(current_dEdy);
        }
        current_dEdy = new_dEdy;

        free(dydx);
        free(dEdx);
        free(dEdW);
        free(dEdb);
    }

    // Free final dEdy after the loop ends
    free(current_dEdy);
}

void Train(struct Layer layer, int epoch)
{
    for (int i = 0; i < epoch; i++)
    {
        printf("Epoch %d, Loss: nan, Acc: nan\n", epoch);
    }
}

void Compute_dEdy(float *dEdy, float *probs, float *y_true, int batchSize, int nClasses)
{
    int total = batchSize * nClasses;
    for (int i = 0; i < total; i++)
    {
        dEdy[i] = (probs[i] - y_true[i]) / batchSize;
        // printf("<%lf>", dEdy[i]);
    }
}

void Normalize(float *normalizedImg, unsigned char *images, int len)
{
    for (int i = 0; i < len; i++)
    {
        normalizedImg[i] = images[i] / 255.0f; // Normalize to 0.0 - 1.0
    }
}

void OneHotEncode(float *onehotLabels, unsigned char *labels, int len, int numClasses)
{
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < numClasses; j++)
        {
            onehotLabels[i * numClasses + j] = (labels[i] == j) ? 1.0f : 0.0f;
        }
    }
    // for (int i = 0; i < numClasses * 20; i++)
    // {
    //     printf("%.0f ", onehotLabels[i]);
    //     if ((i + 1) % numClasses == 0)
    //         printf("\n");
    // }
}

int main(int argc, char const *argv[])
{

    const char *image_path = "/app/dataset/MNIST/train-images-idx3-ubyte";
    const char *label_path = "/app/dataset/MNIST/train-labels-idx1-ubyte";

    int num_images = 0, numLabels = 0;
    unsigned char *images = load_mnist_images(image_path, &num_images);
    unsigned char *labels = load_mnist_labels(label_path, &numLabels);

    if (num_images != numLabels)
    {
        printf("Mismatch between images and labels count.\n");
        exit(1);
    }

    printf("Label[0] = %d\n", labels[0]);
    // what
    float *normalizedImg = calloc(28 * 28 * num_images, sizeof(float));
    float *onehotLabels = calloc(10 * numLabels, sizeof(float));
    Normalize(normalizedImg, images, 28 * 28 * num_images);
    OneHotEncode(onehotLabels, labels, numLabels, 10);
    // for (int i = 0; i < 28 * 28; i++)
    // {
    //     if (normalizedImg[i] > 0)
    //     {
    //         printf("# ", normalizedImg[i] * 255);
    //     }
    //     else
    //     {
    //         printf("- ", normalizedImg[i]);
    //     }
    //     if ((i + 1) % 28 == 0)
    //         printf("\n");
    // }

    float *X = normalizedImg;
    float *y = onehotLabels;

    srand(time(NULL));

    // (input connection, neuron count)
    struct Layer layerSequence[] = {
        {.inFeature = 784, .outFeature = 32, .activation = RELU},
        {.inFeature = 32, .outFeature = 10, .activation = NONE}};
    int numLayers = sizeof(layerSequence) / sizeof(layerSequence[0]);
    printf("there are %d layers in this network\n", numLayers);

    // note:weight is transposed already
    for (int i = 0; i < numLayers; i++)
    {
        initWeights(&layerSequence[i]);
    }
    float loss = 0;
    int numClasses = 10;
    int batchSize = 2400;
    float *logits = calloc(numClasses * numLabels, sizeof(float));
    float *dEdy = calloc(batchSize * numClasses, sizeof(float));
    for (int i = 0; i < 37; i++)
    {
        /* code */
        ForwardProp(logits, layerSequence, numLayers, X, batchSize);
        // softmax_batch(logits, batchSize, numLayers - 1);
        Compute_dEdy(dEdy, logits, y, batchSize, numClasses);
        // backrpop problem
        BackwardProp(layerSequence, dEdy, numLayers, batchSize, 0.1);
        loss = CalculateCCELoss(logits, y, batchSize, numClasses);
        printf("loss: %lf\n", loss);
    }
    // Print result
    // for (int i = 0; i < 3; i++)
    // {
    //     for (int j = 0; j < 3; j++)
    //     {
    //         printf("%.4f ", dEdy[i][j]);
    //     }
    //     printf("\n");
    // }

    free(logits);
    free(dEdy);
    free(images);
    free(labels);
    free(normalizedImg);
    free(onehotLabels);
    printf("end program\n");
    return 0;
}
