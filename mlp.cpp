/*
 _____ __    _____
|     |  |  |  _  |
| | | |  |__|   __|
|_|_|_|_____|__|    C++11 Standard

Multilayer perceptron (MLP) with backpropagation
A very simple implementation of artificial neural network

Written by Emilio Schinina' (emilioschi@gmail.com), JAN 2019

File:	mlp.cpp
*/

#include "mlp.h"

// TRY:
// sig(x) = 2 / (1 + exp(-x)) - 1;
// sig(x) = 2 / (1/2^x + 1) - 1

// TODO: ADD threshold precision parameter

/* It maps the resulting values into the desired range */
double sigmoid(double x) {
	if (x < -3.0) return 0.0;
	if (x > 3.0) return 1.0;
	return tanh(x);
}

/* Derivative of sigmoid function */
double deriv_sigmoid(double x) {
	if (fabs(x) < 0.3) return 1.0;
	if (fabs(x) > 3) return 0.0;
	return (1 / pow(cosh(x), 2));
}

mlp::mlp(int layer, int node, int inputs, int outputs)
{
	ASSERT(layer >= 3 && node >= 2);
	ASSERT(inputs >= 1 && outputs >= 1);

	_NUM_LAYER_	 = layer;
	_NUM_NODE_	 = node;
	_NUM_INPUT_	 = inputs;
	_NUM_OUTPUT_	 = outputs;

	randseed();

	/* Creation of weights matrix */
	weight.push_back(matrix(_NUM_INPUT_, _NUM_NODE_));
	for (int i = 1; i < layer - 2; i++)
		weight.push_back(matrix(_NUM_NODE_, _NUM_NODE_));
	weight.push_back(matrix(_NUM_NODE_, _NUM_OUTPUT_));

	/* Creation of biases matrix */
	for (int i = 0; i < _NUM_LAYER_ - 2; i++)
		bias.push_back(matrix(1, _NUM_NODE_));
	bias.push_back(matrix(1, _NUM_OUTPUT_));


	/* -------------------------
	Initial condition are very important.
	MLP solution is obtained if the first
	layers will have small value than the
	next layers.
	*/
	double scale;
	double rand_value;

	scale = (double)_MAX_RAND_W_ / (_NUM_LAYER_ - 1);
	rand_value = 0;
	for (int i = 0; i < _NUM_LAYER_ - 1; i++){
		rand_value += scale;
		weight[i].randmatrix(-rand_value, rand_value); // dipende dalla activ funct
	}

	scale = (double)_MAX_RAND_B_ / (_NUM_LAYER_ - 1);
	rand_value = 0;
	for (int i = 0; i < _NUM_LAYER_ - 1; i++) {
		rand_value += scale;
		bias[i].randmatrix(-rand_value, rand_value); // dipende dalla activ funct
	}
	/* ------------------------- */


	#if _DEBUG_W_
		printf("[NET]  ---- Random weights & biases ----\n");
		for (int i = 0; i < _NUM_LAYER_ - 1; i++) {
			weight[i].print("W RAND");
			bias[i].print("B RAND");
		}
		printf("       ---------------------------------\n");
	#endif
}

mlp::~mlp()
{
	#if _DEBUG_W_
		printf("[NET] ---- Final matrix ----\n");
		for (int i = 0; i < _NUM_LAYER_ - 1; i++) {
			weight[i].print("W FINAL");
			bias[i].print("B FINAL");
		}
		printf("      -----------------------\n");
	#endif
}

matrix mlp::matrixsigmoid(matrix b)
{
	matrix result(b.getheight(), b.getwidth());

	double value;
	for (int i = 1; i <= b.getheight(); i++) {
		for (int j = 1; j <= b.getwidth(); j++) {
			value = sigmoid(b.getvalue(i, j));
			result.setvalue(i, j, value);
		}
	}
	return result;
}

matrix mlp::matrixsigmoidderivative(matrix b)
{
	matrix result(b.getheight(), b.getwidth());

	double value;
	for (int i = 1; i <= b.getheight(); i++) {
		for (int j = 1; j <= b.getwidth(); j++) {
			value = deriv_sigmoid(b.getvalue(i, j));
			result.setvalue(i, j, value);
		}
	}
	return result;
}


/*
Forward phase

H1 = sig(X * W1)

foreach hidden layer
  H2 = sig(H1 * W2)
  H3 = sig(H2 * W3)
  ...
  Hn-1 = sig(Hn-2 * Wn-1)

Hn = sig(Hn-1 * Wn)
*/

matrix mlp::solve(matrix input)
{
	matrix solution;

	H.push_back(input);
	for (int layer = 0; layer < _NUM_LAYER_ - 1; layer++)
		H.push_back(matrixsigmoid(H[layer] * weight[layer] + bias[layer]));

	solution = H[_NUM_LAYER_-1];

	H.erase(H.begin(), H.end()); /* Clear memory */

	return solution;
}

/* Backpropagation implementation */
double mlp::learn(matrix input, matrix output)
{
	matrix X, Y;
	matrix gradient_w, gradient_b;
	matrix index(1, input.getheight());
	int ii;

	double error = 1; /* Set big for first iteration */

	for (int i = 1;  i <= _MAX_ITERATION_ && error > _TOLLERANCE_; i++) {
		#if _DEBUG_MLP_
			printf("[NET] Learn iteration: %d\n", i);
		#endif

		error = 0; /* Set to 0 for next iteration */

		index.randindex(); /* disorder data index creation */

		/* foreach cluster input */
		for (int e = 1; e <= input.getheight(); e++) {
			/* Set input and output entry */
			ii = (int)index.getvalue(1, e); /* Get index */

			/* Get ii-th row of input and output */
			X = input.strip_row(ii);
			Y = output.strip_row(ii);

			/* --- Forward phase --- */
			H.push_back(X);
			Z.push_back(X);
			for (int layer = 0; layer < _NUM_LAYER_ - 1; layer++) {
				Z.push_back(H[layer] * weight[layer]  + bias[layer]);
				H.push_back(matrixsigmoid(H[layer] * weight[layer]  + bias[layer]));
			}

			/* --- Error calculation --- */
			for (int c = 1; c <= output.getwidth(); c++)
				error += fabs(H[_NUM_LAYER_-1].getvalue(1, c) - Y.getvalue(1, c));

			/* --- Backward phase --- */
			for (int layer = _NUM_LAYER_ - 1, r = 0; layer > 0; layer--, r++) {
				dNETpi_dW.push_back(H[layer-1].transpose());
				dOpi_dNETpi.push_back(matrixsigmoidderivative(H[layer]));

				/* dE_dOpi calculation */
				if (r == 0) {
					// derivative((solution - Y)^2)
					dE_dOpi.push_back(H[_NUM_LAYER_ - 1] - Y);
					//dE_dOpi[r].scale(2.0); // derivative((solution - Y)^2 / 2)
				} else if (r == 1) {
					dE_dOpi[r-1].dot(matrixsigmoidderivative(Z[_NUM_LAYER_ - 1])); // 2*(Hn-Y).*dsig(Hn)

					dE_dOpi.push_back(dE_dOpi[r-1] * (weight[_NUM_LAYER_ - 2].transpose()));
				} else {
					dE_dOpi[r-1].dot(matrixsigmoidderivative(Z[layer + 1]));
					dE_dOpi.push_back(dE_dOpi[r-1] * weight[layer].transpose());
				}

				dE_dOpi[r].dot(dOpi_dNETpi[r]);
				gradient.push_back(dNETpi_dW[r]*dE_dOpi[r]);
			}

			/* --- Weight update --- */
			for (int c = _NUM_LAYER_ - 2, r = 0; c >= 0; c--, r++) {

				#if _DEBUG_MLP_
					weight[c].print("Old weight");
				#endif

				gradient_w = gradient[r] * (double)_LEARNING_RATE_W_;
				gradient_b = gradient[r].median_column() * (double)_LEARNING_RATE_B_;
				weight[c] = weight[c] - gradient_w;
				bias[c] = bias[c] - gradient_b;

				#if _DEBUG_MLP_
					gradient_w.print("gradient W");
					gradient_b.print("gradient B");
					weight[c].print("New weight");
				#endif
			}

			/* Erase old matrix */
			gradient.erase(gradient.begin(), gradient.end());
			dNETpi_dW.erase(dNETpi_dW.begin(), dNETpi_dW.end());
			dOpi_dNETpi.erase(dOpi_dNETpi.begin(), dOpi_dNETpi.end());
			dE_dOpi.erase(dE_dOpi.begin(), dE_dOpi.end());

			Z.erase(Z.begin(), Z.end());
			H.erase(H.begin(), H.end());
		}

		#if _DEBUG_MLP_
			printf("[ERR] Error value: %.15f\n", error);
		#endif

	}
	return error;
}
