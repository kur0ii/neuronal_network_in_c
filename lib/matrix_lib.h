#ifndef __MATRIXLIB__
#define __MATRIXLIB__

/*
 * Structure représentant une matrice
 */
typedef struct {
    int rows ; 
    int columns ;
    double ** data ;
} Matrix ;

/* Description:
 *   Alloue dynamiquement une matrice en fonction du nombre de lignes
 *   et de colonnes spécifiées.
 *
 * Parameters:
 *   rows - Nombre de lignes de la matrice.
 *   columns - Nombre de colonnes de la matrice.
 *
 * Returns:
 *   La matrice allouée.
 */
Matrix allocate_matrix(int rows, int columns);

/* Description:
 *   Libère la mémoire alloué d'une matrice
 *
 * Parameters : 
 *   m1 - Matrice 
 */
void free_matrix(Matrix *mat);

/* Description:
 *   Additionne une matrice avec une matrice de biais.
 *
 * Parameters : 
 *   m1 - La matrice à laquelle ajouter la matrice de biais.
 *   biais - La matrice de biais à ajouter à chaque élément de m1 (doit être un scalaire 1x1).
 *
 * Returns:
 *   La matrice resultante de l'addition.
 */
Matrix add_matrix(Matrix mat1, Matrix mat2);

/* Description:
 *   Soustrait une matrice avec une matrice de biais.
 *
 * Parameters : 
 *   m1 - Matrice
 *   m2 - Matrice
 *
 * Returns:
 *   La matrice resultante de la soustraction.
 */
Matrix sub_matrix(Matrix mat1, Matrix mat2);

/* Description:
 *   Multiplie 2 matrices.
 *
 * Parameters : 
 *   m1 - Matrice
 *   m2 - Matrice
 * Returns:
 *   La matrice resultante de la multiplication.
 */
Matrix multiply_matrix(Matrix mat1, Matrix mat2);


Matrix normal_multiply_matrix(Matrix mat1, Matrix mat2);

/* Description:
 *   Applique une fonction à tous les floatants de la matrice.
 *
 * Parameters : 
 *   m1 - Matrice
 *   func - fonction prenant en paramètre un double et retounant un double
 *
 * Returns:
 *   La matrice resultante de l'application de la fonction func.
 */
Matrix map_matrix(Matrix mat, double(*func)(double));

Matrix scalar_multiply_matrix(Matrix mat, double scalar);

Matrix scalar_sub_matrix(Matrix mat, double scalar);

Matrix scalar_add_matrix(Matrix mat, double scalar);

/* Fonctions à utiliser avec map_matrix */
double invert(double x);
double increment(double x);
double exponential(double x);
double negate(double x);
double one_minus_x(double x);

Matrix sumRows(Matrix mat);

Matrix transpose_matrix(Matrix mat);


/* Description:
 *   Génere un nombre aléatoire tiré d'une distribution normale standard
 *   
 * Returns:
 *   Le nombre aléatoire.
 */
double randn();

/* Description:
 *   Modifie la matrice passé en paramètre et la rempli de nombre aléatoire
 *   grâce à randn().
 *
 * Parameters : 
 *   m1 - Matrice   
 */
void randn_matrix(Matrix *matrix);

double sum_matrix(Matrix mat);

void print_matrix(Matrix mat, char * name);

#endif 