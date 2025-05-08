#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// https://github.com/cvdfoundation/mnist?tab=readme-ov-file

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

void matmul_batch(float *out, float *X, float *W, int batchSize, int inF, int outF)
{
    // printf("\n\n======= matmul (batched) =======\n\n");

    for (int b = 0; b < batchSize; ++b)
    {
        float *x = &X[b * inF];
        float *xout = &out[b * outF];

        for (int i = 0; i < outF; ++i)
        {
            float val = 0.0f;
            for (int j = 0; j < inF; ++j)
            {
                // W[i][j] * x[j]
                val += W[i * inF + j] * x[j];
            }
            xout[i] = val;
            // printf("%.2f ", val);
        }
        // printf("\n");
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
    // for (int i = 0; i < batchSize; ++i)
    // {
    //     for (int j = 0; j < layerSequence[0].inFeature; ++j)
    //     {
    //         printf("%.2f ", input[i * layerSequence[0].inFeature + j]);
    //     }
    //     printf("\n\n");
    // }
    // FIX maybe
    float temp1[256]; // scratch buffer 1
    float temp2[256]; // scratch buffer 2
    float *current = input;
    float *next = temp1;
    for (int layer = 0; layer < 2; layer++)
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
        memcpy(layerSequence[layer].z, next, batchSize * outF * sizeof(float));
        layerSequence[layer].X = calloc(batchSize * inF, sizeof(float));
        memcpy(layerSequence[layer].X, current, batchSize * inF * sizeof(float));
        // if (layerSequence->activation)
        // {

        //     /* code */
        // }

        float *temp = current;
        current = next;
        if (layer < 2 - 1)
        {
            next = (temp == temp1) ? temp2 : temp1;
        }
    }
    softmax_batch(next, layerSequence[2 - 1].outFeature, batchSize);

    // printf("\n\nafter softmax\n\n");
    // for (int i = 0; i < batchSize; i++)
    // {
    //     /* code */
    //     for (int j = 0; j < layerSequence[1].outFeature; j++)
    //     {
    //         printf("%.2lf ", next[i * layerSequence[1].outFeature + j]);
    //     }
    //     printf("\n\n");
    // }

    // plain copy value
    memcpy(logits, next, batchSize * layerSequence[2 - 1].outFeature * sizeof(float));
}
void BackwardProp(struct Layer *layerSequence, float *dEdy, int numLayers, int batchSize, float learningRate)
{
    for (int layer = numLayers - 1; layer >= 0; layer--)
    {
        int inF = layerSequence[layer].inFeature;
        int outF = layerSequence[layer].outFeature;
        // just points to the address
        float *W = layerSequence[layer].Weight;
        float *X = layerSequence[layer].X;
        float *z = layerSequence[layer].z;

        // printf("BackpropLayer %d (%d → %d):\n", layer, inF, outF);
        // batch, outF
        float *dydx = malloc(batchSize * outF * sizeof(float));
        // batch, outF - same because its a dot product of dydx
        float *dEdx = malloc(batchSize * outF * sizeof(float));
        // inf, outf - because it is ther result of X.T @ dedx, same as weight
        float *dEdW = calloc(inF * outF, sizeof(float));
        // outf - one dim only, unititalized because we dont have weigths yet
        float *dEdb = calloc(outF, sizeof(float));

        for (int i = 0; i < batchSize * outF; i++)
        {
            // ReLU: derivative is 1 if z > 0
            // dydx[i] = z[i] > 0 ? 1.0f : 0.0f;
            // Plain: derivative is 1
            dydx[i] = 1.0f;
            dEdx[i] = dEdy[i] * dydx[i];
            // printf(" %.2lf ", dEdx[i]);
        }

        // dEdW = X^T * dEdx
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inF; i++)
            {
                for (int j = 0; j < outF; j++)
                {
                    dEdW[j * inF + i] += X[b * inF + i] * dEdx[b * outF + j];
                    // printf(" %.2lf ", dEdW[j * inF + i]);
                }
            }
        }

        // dEdB is sum of dEdx over batch
        // for (int j = 0; j < outF; j++)
        // {
        //     for (int b = 0; b < batchSize; b++)
        //     {
        //         dEdb[j] += dEdx[b * outF + j];
        //         // printf(" %.2lf ", dEdb[j]);
        //     }
        // }

        // update weight
        // remember the size is inxouts
        for (int i = 0; i < inF * outF; i++)
        {
            W[i] -= learningRate * dEdW[i] / batchSize;
            // printf(" %.2lf ", W[i]);
        }

        // no bias yet, comment this
        // for (int j = 0; j < outF; j++)
        // {
        //     layerSequence[layer].Bias[j] -= learningRate * dEdb[j] / batchSize;
        // }

        // // Compute dEdy for previous layer (dEdx @ W^T)
        // printf("out %d \n", layer);

        float *new_dEdy = malloc(batchSize * inF * sizeof(float));
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inF; i++)
            {
                float sum = 0;
                for (int j = 0; j < outF; j++)
                {
                    sum += dEdx[b * outF + j] * W[j * inF + i];
                }
                new_dEdy[b * inF + i] = sum;
            }
        }

        dEdy = new_dEdy;

        // // Free memory
        free(dydx);
        free(dEdx);
        free(dEdW);
        free(dEdb);
    }
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

int main(int argc, char const *argv[])
{
    float X[3][30] = {{0, 0, 1, 1, 0, 0,
                       0, 1, 0, 0, 1, 0,
                       1, 1, 1, 1, 1, 1,
                       1, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 1},
                      {0, 1, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 0,
                       0, 1, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 0,
                       0, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 0,
                       0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0,
                       0, 1, 1, 1, 1, 0}};

    float y[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    srand(time(NULL));

    // (input connection, neuron count)
    struct Layer layerSequence[] = {
        {.inFeature = 30, .outFeature = 32, .activation = RELU},
        {.inFeature = 32, .outFeature = 3, .activation = NONE}};
    int numLayers = sizeof(layerSequence) / sizeof(layerSequence[0]);
    printf("%d", numLayers);

    // note:weight is transposed already
    for (int i = 0; i < numLayers; i++)
    {
        initWeights(&layerSequence[i]);
    }
    float loss = 0;
    // 3 pic, 3 batch
    float logits[3][3];
    float dEdy[3][3];
    for (int i = 0; i < 20; i++)
    {
        /* code */
        ForwardProp(&logits[0][0], layerSequence, numLayers, &X[0][0], 3);
        // still broken
        // softmax_batch(&logits[0][0], numLayers - 1, 3);
        Compute_dEdy(&dEdy[0][0], &logits[0][0], &y[0][0], 3, 3);
        BackwardProp(layerSequence, &dEdy[0][0], numLayers, 3, 0.1);
        loss = CalculateCCELoss(&logits[0][0], &y[0][0], 3, 3);
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

    printf("end program\n");
    return 0;
}
