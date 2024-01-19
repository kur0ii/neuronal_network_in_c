#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_lib.h"

#ifndef __NEURONALLIB__
#define __NEURONALLIB__

typedef struct Layer {
    int nb_neurons;
    Matrix W;
    Matrix b;
    Matrix Z;
    Matrix A;
    struct Layer *next;
} Layer;

typedef struct {
    Layer *head;
} LayerList;


void logical_gate_init(Matrix *X, Matrix *y_and, Matrix *y_or, Matrix *y_xor);


/* Description:
 *   Initialise la matrice poids et biais aléatoirement grâce à randn().
 *
 * Parameters : 
 *   X - Matrice features
 *   W - Matrice des poids
 *   b - Matrice biais   
 */
void initialization(Matrix X, Matrix *W, Matrix *b);



/* Description:
 *   Initialise la matrice poids et biais aléatoirement grâce à randn().
 *
 * Parameters : 
 *   X - Matrice features
 *   W - Matrice des poids
 *   b - Matrice biais  
 *
 * Returns :
 *   La matrice des probabilités
 */
Matrix model(Matrix X, Matrix W, Matrix b);


/* Description:
 *   Calcul le logloss.
 *
 * Parameters : 
 *   A - Matrice des probabilités
 *   y - Matrice target 
 */
double log_loss(Matrix A, Matrix y);

/* à corriger */

void gradients(Matrix A, Matrix X, Matrix y, Matrix *dW, Matrix *db);

void update(Matrix *W, Matrix *b, Matrix dW, Matrix db, const double learning_rate);

Matrix predict(Matrix X, Matrix W, Matrix b);

void artificial_neuron(Matrix X, Matrix y, const double learning_rate, const int n_iter);

/* 2 couches*/

void initialization_2(LayerList **layers, int nb_entry);
void forward_propagation(Matrix X, LayerList *layers);
void back_propagation(Matrix X, Matrix y, LayerList *layers, Matrix *dW1, Matrix *db1, Matrix *dW2, Matrix *db2);
void update_2(Matrix dW1, Matrix dW2, Matrix db1, Matrix db2, LayerList *layers, double learning_rate);
Matrix predict_2(const LayerList *layers);
void artificial_neuron_2(Matrix X, Matrix y, double learning_rate, int n_iter);
#endif