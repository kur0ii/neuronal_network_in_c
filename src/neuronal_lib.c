#include "../lib/neuronal_lib.h"

void logical_gate_init(Matrix *X, Matrix *y_and, Matrix *y_or, Matrix *y_xor) {
    double X_comb[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    // Create matrices to store the data
    *X = allocate_matrix(2, 4);
    *y_and = allocate_matrix(1, 4);
    *y_or = allocate_matrix(1, 4);
    *y_xor = allocate_matrix(1, 4);

    // Fill the matrix X with the input values
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            X->data[i][j] = X_comb[i][j];
        }
    }

    for (int j = 0; j < 4; ++j) {
        y_and->data[0][j] = X_comb[0][j] && X_comb[1][j];
        y_or->data[0][j] = X_comb[0][j] || X_comb[1][j];
        y_xor->data[0][j] = (X_comb[0][j] || X_comb[1][j]) && !(X_comb[0][j] && X_comb[1][j]);

    }
}




void initialization_2(LayerList **layers, int nb_entry) {

    *layers = malloc(sizeof(LayerList));

    // Allouer la couche cachée 1
    Layer *layer1 = malloc(sizeof(Layer));
    layer1->nb_neurons = 2; // n1
    layer1->W = allocate_matrix(layer1->nb_neurons, nb_entry);
    layer1->b = allocate_matrix(layer1->nb_neurons, 1);
    randn_matrix(&layer1->W);
    randn_matrix(&layer1->b);

    // Allouer la couche cachée 2
    Layer *layer2 = malloc(sizeof(Layer));
    layer2->nb_neurons = 1 ; // Taille des entrées
    layer2->W = allocate_matrix(layer2->nb_neurons, layer1->nb_neurons);
    layer2->b = allocate_matrix(layer2->nb_neurons, 1);
    randn_matrix(&layer2->W);
    randn_matrix(&layer2->b);

    // Ajouter les couches à la liste
    layer1->next = layer2;
    (*layers)->head = layer1;
}


void initialization(Matrix X, Matrix *W, Matrix *b){
    *W = allocate_matrix(X.columns,1);
    *b = allocate_matrix(1,1);
    randn_matrix(W);
    randn_matrix(b); 
}

void forward_propagation(Matrix X, LayerList *layers){
    Matrix Z, Z_neg, exp_Z_neg, increment_exp_Z_neg, X_dot_W  ; 

    Layer *layer1 ;
    Layer *layer2 ; 

    layer1 = layers->head ; 
    layer2 = layer1->next ; 
    
    Z = scalar_add_matrix(multiply_matrix(layer1->W,X),layer1->b.data[0][0]) ; 
    layer1->A = map_matrix(map_matrix(map_matrix(map_matrix(Z,negate),exp), increment), invert) ;

    /* A2 */
    X_dot_W = multiply_matrix(layer2->W,layer1->A);
    Z = scalar_add_matrix(X_dot_W,layer2->b.data[0][0]) ;
    Z_neg = map_matrix(Z,negate); 
    exp_Z_neg = map_matrix(Z_neg,exp);  
    increment_exp_Z_neg = map_matrix(exp_Z_neg, increment);
    layer2->A = map_matrix(increment_exp_Z_neg, invert) ;
}


void back_propagation(Matrix X, Matrix y, LayerList *layers, Matrix *dW1, Matrix *db1, Matrix *dW2, Matrix *db2){
    Matrix dZ1, dZ2, aT, xT, dZ2_dot_w2T,  one_minus_A1,A1_dot_dZ2_dot_w2T ; 
    double m ;
    Layer *layer1 ;
    Layer *layer2 ; 

    layer1 = layers->head ; 
    layer2 = layer1->next ; 

    m = (double)y.columns;
    
    // d2 

    dZ2 = sub_matrix(layer2->A,y);
    aT = transpose_matrix(layer1->A);
    *dW2 = scalar_multiply_matrix(multiply_matrix(dZ2,aT),1/m);
    *db2 = scalar_multiply_matrix(sumRows(dZ2),1 / m);


    // d1 
    dZ2_dot_w2T = multiply_matrix(transpose_matrix(layer2->W),dZ2);
    A1_dot_dZ2_dot_w2T = normal_multiply_matrix(layer1->A,dZ2_dot_w2T);
    // fprintf(stderr,"still alive ..\n");
    // print_matrix(dZ2_dot_w2T,"dZ2_dot_w2T");
    // print_matrix(layer1->A, "A1");
    // print_matrix(A1_dot_dZ2_dot_w2T,"A1_dot_dZ2_dot_w2T");
    one_minus_A1 = map_matrix(layer1->A,one_minus_x);
    // print_matrix(one_minus_A1,"one_minus_A1");
    dZ1 = normal_multiply_matrix(A1_dot_dZ2_dot_w2T,one_minus_A1);

    
    // fprintf(stderr,"still alive ..\n");
    xT = transpose_matrix(X);
    //fprintf(stderr,"still alive ..\n");
    *dW1 = scalar_multiply_matrix(multiply_matrix(dZ1,xT),1/m);
    print_matrix(*dW1,"dW1");
    *db1 = scalar_multiply_matrix(sumRows(dZ1),1 / m);
}


void update_2(Matrix dW1, Matrix dW2, Matrix db1, Matrix db2, LayerList *layers, double learning_rate){
    Layer *layer1 ;
    Layer *layer2 ; 

    layer1 = layers->head ; 
    layer2 = layer1->next ; 

    print_matrix(dW1,"dw1"); // 2x2 ok
    print_matrix(db1,"db1"); // 2x1 ok
    print_matrix(dW2,"dw2"); // 1x2 ok
    print_matrix(db2,"db2"); // 1x1 ok

    print_matrix(layer2->W, "W2");
    print_matrix(scalar_multiply_matrix(db2,-learning_rate), "db2.learn");

    layer1->W = sub_matrix(layer1->W,scalar_multiply_matrix(dW1,-learning_rate));
    fprintf(stderr,"ici 1\n");
    layer1->b = sub_matrix(layer1->b,scalar_multiply_matrix(db1,-learning_rate));
    fprintf(stderr,"ici 2\n");
    layer2->W = sub_matrix(layer2->W,scalar_multiply_matrix(dW2,-learning_rate));
    fprintf(stderr,"ici 3\n");
    layer2->b = scalar_sub_matrix(layer2->W,scalar_multiply_matrix(db2,-learning_rate).data[0][0]);
    fprintf(stderr,"ici 4\n");

}

Matrix predict_2(const LayerList *layers) {
    Matrix result ; 

    Layer *layer1 ;
    Layer *layer2 ; 

    layer1 = layers->head ; 
    layer2 = layer1->next ; 

    // Create a new matrix for the result
    result = allocate_matrix(layer2->A.rows, layer2->A.columns);

    // Check if each element of A is greater than or equal to 0.5
    for (int i = 0; i < layer2->A.rows; ++i) {
        for (int j = 0; j < layer2->A.columns; ++j) {
            printf("predict value : %f\n",layer2->A.data[i][j]);
            result.data[i][j] = (layer2->A.data[i][j] >= 0.5) ? 1.0 : 0.0;
        }
    }

    return result;
}

Matrix model(Matrix X, Matrix W, Matrix b){
    Matrix Z, Z_neg, exp_Z_neg, A, increment_exp_Z_neg, X_dot_W  ; 


    X_dot_W = multiply_matrix(X,W);
    Z = scalar_add_matrix(X_dot_W, b.data[0][0]) ;
    Z_neg = map_matrix(Z,negate);
    exp_Z_neg = map_matrix(Z_neg,exp);  
    increment_exp_Z_neg = map_matrix(exp_Z_neg, increment);
    A = map_matrix(increment_exp_Z_neg, invert) ;

    return A ; 
}


double log_loss(Matrix A, Matrix y) {
    double loss ; 
    if (A.rows != y.rows || A.columns != 1 || y.columns != 1) {
        // Error: Matrices have incompatible dimensions
        printf("Error: Incompatible dimensions for log_loss calculation.\n");
        exit(EXIT_FAILURE);
    }

    loss = 0.0;

    for (int i = 0; i < A.rows; ++i) {
        loss += -y.data[i][0] * log(A.data[i][0]) - (1 - y.data[i][0]) * log(1 - A.data[i][0]);
    }

    return 1.0 / A.rows * loss;
}


void gradients(Matrix A, Matrix X, Matrix y, Matrix *dW, Matrix *db) {
    Matrix sub_a_y, transpose_X, mult ; 

    if (A.rows != y.rows || A.columns != 1 || y.columns != 1 || X.rows != A.rows || X.columns != 2) {
        // Error: Matrices have incompatible dimensions
        printf("Error: Incompatible dimensions for gradients calculation.\n");
        exit(EXIT_FAILURE);
    }

    *dW = allocate_matrix(X.columns, 1);
    *db = allocate_matrix(1, 1);

    sub_a_y = sub_matrix(A,y) ; 
    transpose_X = transpose_matrix(X);
    mult = multiply_matrix(transpose_X, sub_a_y);
    *dW = scalar_multiply_matrix(mult, 1.0 / y.rows);
    db->data[0][0] = 1.0 / A.rows * sum_matrix(sub_a_y);
}

void update(Matrix *W, Matrix *b, Matrix dW, Matrix db, const double learning_rate) {
    Matrix sub ; 

    if (W->rows != dW.rows || W->columns != dW.columns || b->rows != db.rows || b->columns != db.columns) {
        // Error: Matrices have incompatible dimensions
        printf("Error: Incompatible dimensions for update operation.\n");
        exit(EXIT_FAILURE);
    }
    sub = scalar_multiply_matrix(dW,learning_rate);
    *W = sub_matrix(*W,sub);

    sub = scalar_multiply_matrix(db,learning_rate);
    *b = sub_matrix(*b,sub);
}

Matrix predict(Matrix X, Matrix W, Matrix b) {
    Matrix A, result ; 

    A = model(X, W, b);

    // Create a new matrix for the result
    result = allocate_matrix(A.rows, A.columns);

    // Check if each element of A is greater than or equal to 0.5
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.columns; ++j) {
            printf("predict value : %f\n",A.data[i][j]);
            result.data[i][j] = (A.data[i][j] >= 0.5) ? 1.0 : 0.0;
        }
    }

    return result;
}

void artificial_neuron(Matrix X, Matrix y, const double learning_rate, const int n_iter) {
    Matrix W, b, dW, db, A, y_pred  ;
    initialization(X, &W, &b);

    for (int i = 0; i < n_iter; ++i) {
        A = model(X, W, b);
        gradients(A, X, y,&dW,&db);
        update(&W, &b, dW, db, learning_rate);
    }

    // Now you can use the trained parameters W and b for prediction
    y_pred = predict(X, W, b);
    print_matrix(y_pred,"y_pred");

    // Free memory
    free_matrix(&W);
    free_matrix(&b);
}

void artificial_neuron_2(Matrix X, Matrix y, double learning_rate, int n_iter){
    LayerList *layers = NULL  ; 
    Matrix dW1, db1, dW2, db2, y_pred ; 
    initialization_2(&layers,2);
    print_matrix(layers->head->W,"W1");
    fprintf(stderr,"Init OK\n");
    db1 = allocate_matrix(1,1);
    db2 = allocate_matrix(1,1);
    for (int i=0; i < n_iter; ++i){
        forward_propagation(X,layers);
        fprintf(stderr,"forward propagration OK\n");
        back_propagation(X,y,layers,&dW1, &db1, &dW2, &db2);
        fprintf(stderr,"back propagration OK\n");
        update_2(dW1,dW2,db1,db2,layers,learning_rate);
        fprintf(stderr,"update OK\n");
    }

    y_pred = predict_2(layers);
    print_matrix(y_pred,"y_pred");

}