/*
 _____ __    _____ 
|     |  |  |  _  |
| | | |  |__|   __|
|_|_|_|_____|__|    C++11 Standard
 
Multilayer perceptron (MLP) with backpropagation
A very simple implementation of artificial neural network

Written by Emilio Schinina' (emilioschi@gmail.com), JAN 2019

File:	matrix.h
*/


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

#define _DEBUG_MATRIX_ false

#define ARRAY(...) ((double []){__VA_ARGS__})

class matrix {

private:
	double* data;
	int row;
	int column;
	int length;
public:

	// Constructor
	matrix(int r = 0, int c = 0);
	matrix(int r, int c, double* array);
	//matrix(char*); // file2matrix

	// Copy Constructor
	matrix(const matrix& source);

	matrix operator+(const matrix& source);
	matrix operator+(double value);
	matrix operator-(const matrix& source);
	matrix operator*(const matrix& source);
	matrix operator*(double scale);

	// Overloaded Assignment
	matrix& operator=(const matrix& source);

	// Destructor
	~matrix();
	
	void print(std::string str = "");

	void randmatrix(double, double);

	void scale(double);
	void dot(matrix);
	matrix transpose();

	double getvalue(int, int);
	void setvalue(int, int, double);
	int getheight () {return row;}
	int getwidth () {return column;}

	matrix strip_row(int i);
	matrix strip_column(int i);
	matrix row_add();
	matrix column_add();


	matrix bubblesort();
	void randindex();
	double median();
	matrix median_column();
	//matrix strip_column(int n);

//	int getnrow();
//	int getncolumn();
};



#endif /* _MATRIX_H_ */
