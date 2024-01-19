#include <stdio.h>
#include "../lib/neuronal_lib.h"

int main(){
    Matrix X,y_and,y_or,y_xor;
    logical_gate_init(&X,&y_and,&y_or,&y_xor);
    print_matrix(X,"X_original");
    //artificial_neuron(X,y_and,0.1,100000);
    artificial_neuron_2(X,y_and,0.1,100000);

    return 0 ; 
}   