/*
 _____ __    _____ 
|     |  |  |  _  |
| | | |  |__|   __|
|_|_|_|_____|__|    C++11 Standard
 
Multilayer perceptron (MLP) with backpropagation
A simple implementation of artificial neural network

Written by Emilio Schinina' (emilioschi@gmail.com)

File:	mlp.h
*/

#ifndef _MLP_H_
#define _MLP_H_

#include "matrix.h"
#include "utils.h"

#include <vector>
#include <math.h>

#define	_DEBUG_MLP_		false
#define	_DEBUG_W_		true

#define	_LEARNING_RATE_W_	0.1
#define	_LEARNING_RATE_B_	0.06

#define	_MAX_ITERATION_		300000
#define	_TOLLERANCE_		1.0E-20

#define	_MAX_RAND_W_	2 // TANH limit
#define	_MAX_RAND_B_	2 // TANH limit

class mlp {
public:
	// matrice con il numero di nodi per ogni hidden layer
	// TODO: matrix.cpp add file2matrix()
	mlp(int, int , int, int);
	~mlp();

	double learn(matrix, matrix); // deve ritornare il valore di errore
	matrix solve(matrix);

	matrix matrixsigmoid(matrix b);
	matrix matrixsigmoidderivative(matrix);
//	bool load(char* filename);
//	bool save(const mlp& source);

private:
	matrix structure;

	int _NUM_LAYER_;
	int _NUM_NODE_; // deve cambiare da matrice in matrice
	int _NUM_INPUT_;
	int _NUM_OUTPUT_;

	// cambiare oggetto vettore
	std::vector<matrix> weight;
	std::vector<matrix> bias;

	std::vector<matrix> H;
	std::vector<matrix> Z;

	std::vector<matrix> dNETpi_dW;
	std::vector<matrix> dOpi_dNETpi;
	std::vector<matrix> dE_dOpi;
	std::vector<matrix> gradient; // dE_dw
};

#endif /* _MLP_H_ */