#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "../lib/matrix_lib.h"

Matrix allocate_matrix(int rows, int columns) {
    Matrix mat;
    mat.rows = rows;
    mat.columns = columns;

    mat.data = (double **)malloc(rows * sizeof(double *));
    if (mat.data == NULL) {
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; ++i) {
        mat.data[i] = (double *)malloc(columns * sizeof(double));
        if (mat.data[i] == NULL) {
            exit(EXIT_FAILURE);
        }
    }
    return mat;
}

Matrix add_matrix(Matrix mat1, Matrix mat2) {
    if (mat1.rows != mat2.rows || mat1.columns != mat2.columns) {
        printf("Erreur : Les matrices ne peuvent pas être additionné.\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = allocate_matrix(mat1.rows, mat1.columns);

    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat1.columns; ++j) {
            result.data[i][j] = mat1.data[i][j] + mat2.data[i][j];
        }
    }

    return result;
}

Matrix sub_matrix(Matrix mat1, Matrix mat2) {
    if (mat1.rows != mat2.rows || mat1.columns != mat2.columns) {
        printf("Erreur : Les matrices ne peuvent pas être soustraites.\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = allocate_matrix(mat1.rows, mat1.columns);

    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat1.columns; ++j) {
            result.data[i][j] = mat1.data[i][j] - mat2.data[i][j];
        }
    }

    return result;
}

Matrix sumRows(Matrix mat) {
    Matrix result;
    result.rows = mat.rows;
    result.columns = 1;

    result.data = (double **)malloc(result.rows * sizeof(double *));
    for (int i = 0; i < result.rows; i++) {
        result.data[i] = (double *)malloc(sizeof(double));
        result.data[i][0] = 0.0;

        for (int j = 0; j < mat.columns; j++) {
            result.data[i][0] += mat.data[i][j];
        }
    }

    return result;
}

Matrix normal_multiply_matrix(Matrix mat1, Matrix mat2){
    Matrix result ; 
    if (mat1.columns != mat2.columns || mat1.rows != mat2.rows){
        printf("Erreur : Les matrices sont de dimensions différentes .\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat1.columns; ++j) {
            result.data[i][j] = mat1.data[i][j] * mat2.data[i][j];
        }
    }
    return result ; 
}
Matrix multiply_matrix(Matrix mat1, Matrix mat2) {
    if (mat1.columns != mat2.rows) {
        // Matrices cannot be multiplied
        printf("Error: Incompatible matrices for multiplication\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the result matrix
    Matrix result = allocate_matrix(mat1.rows, mat2.columns);

    // Perform matrix multiplication
    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat2.columns; ++j) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < mat1.columns; ++k) {
                result.data[i][j] += mat1.data[i][k] * mat2.data[k][j];
            }
        }
    }
    return result;
}

Matrix scalar_multiply_matrix(Matrix mat, double scalar) {
    Matrix result = allocate_matrix(mat.rows, mat.columns);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            result.data[i][j] = mat.data[i][j] * scalar;
        }
    }

    return result;
}

Matrix scalar_sub_matrix(Matrix mat, double scalar) {
    Matrix result = allocate_matrix(mat.rows, mat.columns);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            result.data[i][j] = mat.data[i][j] - scalar;
        }
    }

    return result;
}

Matrix scalar_add_matrix(Matrix mat, double scalar) {
    Matrix result = allocate_matrix(mat.rows, mat.columns);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            result.data[i][j] = mat.data[i][j] + scalar;
        }
    }

    return result;
}

double sum_matrix(Matrix mat1){
    double res = 0.0 ; 

    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat1.columns; ++j) {
            res += mat1.data[i][j];
        }
    }
    return res ; 
}

Matrix map_matrix(Matrix mat1, double(*func)(double)) {
    Matrix result = allocate_matrix(mat1.rows, mat1.columns);

    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat1.columns; ++j) {
            result.data[i][j] = func(mat1.data[i][j]);
        }
    }

    return result;
}

Matrix transpose_matrix(Matrix mat1) {
    /* inverse colonne et ligne */
    Matrix result = allocate_matrix(mat1.columns,mat1.rows);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.columns; j++) {
            result.data[j][i] = mat1.data[i][j];
        }
    }
    return result;
}

double invert(double x){
    return 1 / x ; 
}

double increment(double x) {
    return ++x ;
}

double exponential(double x) {
    return exp(x);
}

double negate(double x) {
    return -x;
}

double one_minus_x(double x){
    return 1 - x ; 
}

void free_matrix(Matrix *mat) {
    for (int i = 0; i < mat->rows; ++i) {
        free(mat->data[i]);
    }

    free(mat->data);
    mat->rows = 0;
    mat->columns = 0;
}

void print_matrix(Matrix mat, char*name) {
    printf("\n****************************\n\n");
    printf("The matrix %s data : \n\n",name);
    for (int i = 0; i < mat.rows; ++i) {
        printf(" [");
        for (int j = 0; j < mat.columns; ++j) {
            printf(" %f", mat.data[i][j]);
        }
        printf(" ]\n\n");
    }
    printf("Shape : (%d,%d)\n",mat.rows,mat.columns);
    printf("\n****************************\n\n");
}


double randn() {
    double u1 = rand() / (double)RAND_MAX; // Nombre aléatoire entre 0 et 1
    double u2 = rand() / (double)RAND_MAX; // Un autre nombre aléatoire entre 0 et 1

    // Transformation de Box-Muller pour obtenir deux nombres aléatoires normalement distribués
    // utiliser cos ou sin 
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    // double z0 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

    return z0;
}

void randn_matrix(Matrix *matrix){
    srand(time(NULL));
    for (int i=0 ; i < matrix->rows ; i++){
        for (int j=0 ; j < matrix->columns ; j++){
            matrix->data[i][j] = randn();
        }
    }
}