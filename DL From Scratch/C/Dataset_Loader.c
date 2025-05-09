#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

    int num_labels = 0;
    fread(&num_labels, sizeof(int), 1, f);
    num_labels = __builtin_bswap32(num_labels);

    unsigned char *labels = malloc(num_labels);
    fread(labels, sizeof(unsigned char), num_labels, f);
    fclose(f);

    *number_of_labels = num_labels;
    return labels;
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

int main()
{
    const char *image_path = "/app/dataset/MNIST/train-images-idx3-ubyte";
    const char *label_path = "/app/dataset/MNIST/train-labels-idx1-ubyte";

    int num_images = 0, num_labels = 0;
    unsigned char *images = load_mnist_images(image_path, &num_images);
    unsigned char *labels = load_mnist_labels(label_path, &num_labels);

    if (num_images != num_labels)
    {
        printf("Mismatch between images and labels count.\n");
        exit(1);
    }

    printf("Label[0] = %d\n", labels[0]);
    // what
    float *normalizedImg = calloc(28 * 28 * num_images, sizeof(float));
    float *onehotLabels = calloc(10 * num_labels, sizeof(float));
    Normalize(normalizedImg, images, 28 * 28 * num_images);
    OneHotEncode(onehotLabels, labels, num_labels, 10);
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

    free(images);
    free(labels);
    free(normalizedImg);
    free(onehotLabels);
    return 0;
}
