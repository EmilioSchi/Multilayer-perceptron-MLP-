/*
 _____ __    _____ 
|     |  |  |  _  |
| | | |  |__|   __|
|_|_|_|_____|__|    C++11 Standard
 
Multilayer perceptron (MLP) with backpropagation
A very simple implementation of artificial neural network

Written by Emilio Schinina' (emilioschi@gmail.com), JAN 2019

File:	matrix.cpp
*/

#include "matrix.h"

matrix::matrix(int r, int c)
{
	#if _DEBUG_MATRIX_
		std::cout << "[#] Allocation of new " << r << "x" <<c<< " matrix: "<< this << std::endl;
	#endif

	row	 = r;
	column	 = c;
	length	 = r * c;

	if(!(data = (double*)malloc(sizeof(double) * length))){
		exit(1);
	}

	memset(data, 0.0, length * sizeof(double));
}

matrix::matrix(int r, int c, double* array)
{
	#if _DEBUG_MATRIX_
		std::cout << "[#] Allocation of new " << r << "x" <<c<< " matrix: "<< this << std::endl;
	#endif

	ASSERT(r > 0 && c > 0);

	row	 = r;
	column	 = c;
	length	 = r * c;

	if(!(data = (double*)malloc(sizeof(double) * length))){
		exit(1);
	}

	double *ptr	 = data;
	double *ptr_d	 = array;

	for (int i = 0; i < length; i++) {
		*ptr = *ptr_d;
		ptr++;
		ptr_d++;
	}
	//memset(data, 0.0, length * sizeof(double));
}


matrix::matrix(const matrix& source)
{
	#if _DEBUG_MATRIX_
		std::cout << "[#] Copy Constructor called. ";
		std::cout << source.row << "x" << source.column << " matrix: "<< this << std::endl;	
	#endif

	row	 = source.row;
	column	 = source.column;
	length	 = row * column;

	if(!(data = (double*)malloc(sizeof(double) * length))){
		exit(1);
	}

	double* ptr_a = data;
	double* ptr_b = source.data;

	// TODO Check NULL pointer

	for (int i = 0; i < length; i++)
		*(ptr_a++) = *(ptr_b++);	
}


double matrix::getvalue(int i, int j)
{
	i -= 1; j -=1; // C become from zero
	return *(data + (i * column) + j);
}


void matrix::setvalue(int i, int j, double value)
{
	i -= 1; j -=1; // C become from zero
	*(data + (i * column) + j) = value;	
}


void matrix::randmatrix(double lower, double upper)
{
	double value;
	double* ptr = data;
	for (int i = 0; i < length; i++) {
		value = lower + (double)rand() / (double)RAND_MAX * (upper - lower);
		*(ptr++) = value;
	}
}

void matrix::scale(double value)
{
	double *ptr = data;
	for (int i = 0; i < length; i++)
		*(ptr++) *= value;
}

void matrix::dot(matrix source)
{
	ASSERT(row == source.row && column == source.column);

	double *ptr_a = data;
	double *ptr_b = source.data;

	for (int i = 0; i < length; i++)
		*(ptr_a++) *= *(ptr_b++);
}

matrix matrix::transpose()
{
	matrix result(column, row);

	double* ptr_out = result.data;
	double* ptr = data;

	for (int i = 0; i < row; i++)
	{
		ptr_out = &result.data[i];
		for (int j = 0; j < column; j++)
		{
			*ptr_out = *ptr;
			ptr++;
			ptr_out += row;
		}
	}

	return result;
}

matrix& matrix::operator=(const matrix& source)
{
	#if _DEBUG_MATRIX_
		std::cout << "[#] Overoaded Assignent called. " << source.row << "x" << source.column << " matrix: "<< this << std::endl;
	#endif

	// self assignment check
	if (this == &source) {
		return *this;
	}

	row	 = source.row;
	column	 = source.column;
	length	 = row * column;

	// cambiano le dimensioni
	if(!(data = (double*)realloc(data, sizeof(double) * length))){
		exit(1);
	}

	double* ptr_a = data;
	double* ptr_b = source.data;

	// TODO Check NULL pointer
	for (int i = 0; i < length; i++)
		*(ptr_a++) = *(ptr_b++);

	return *this;
}

matrix matrix::operator+(const matrix& source)
{
	ASSERT(row == source.row && column == source.column);

	matrix result(row, column);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;
	double* ptr_b	 = source.data;

	for (int i = 0; i < length; ++i)
		*(ptr_out++) = *(ptr_a++) + *(ptr_b++);

	return result;
}

matrix matrix::operator-(const matrix& source)
{
	ASSERT(row == source.row && column == source.column);

	matrix result(row, column);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;
	double* ptr_b	 = source.data;

	for (int i = 0; i < length; ++i)
		*(ptr_out++) = *(ptr_a++) - *(ptr_b++);

	return result;
}

matrix matrix::operator*(const matrix& source)
{
	ASSERT(column == source.row);
	ASSERT(column != 0);
	ASSERT(row != 0);

	#if _DEBUG_MATRIX_
		std::cout << "[#] Multiplicity " << row << "x" << source.column << " matrix: "<< this << std::endl;
	#endif

	matrix result(row, source.column);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;
	double* ptr_b	 = source.data;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < source.column; j++) {
			ptr_a = &data[ i * column ];
			ptr_b = &source.data[ j ];
			*ptr_out = 0;
			for (int k = 0; k < column; k++) {
				*ptr_out += *ptr_a * *ptr_b;
				ptr_a++;
				ptr_b += source.column;
			}
		ptr_out++;
		}
	}

	return result;
}

matrix matrix::operator+(double value)
{
	ASSERT(column != 0);
	ASSERT(row != 0);

	#if _DEBUG_MATRIX_
		std::cout << "[#] Add value. value: " << value << ": " << row << "x" << column << " matrix: "<< this << std::endl;
	#endif

	matrix result(row, column);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;

	for (int i = 0; i < length; i++)
		*(ptr_out++) = *(ptr_a++) + value;

	return result;
}

matrix matrix::operator*(double scale)
{
	ASSERT(column != 0);
	ASSERT(row != 0);

	#if _DEBUG_MATRIX_
		std::cout << "[#] Scale multiplicity. value: " << scale << ": " << row << "x" << column << " matrix: "<< this << std::endl;
	#endif

	matrix result(row, column);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;

	for (int i = 0; i < length; i++)
		*(ptr_out++) = *(ptr_a++) * scale;

	return result;
}

void matrix::randindex()
{
	int j,k;
	int n = column;
	double tmp;

	double* ptr = data;

	for (int i = 1; i <= n; i++)
		*(ptr++) = i;

	for (int i = 0; i < n * 2; i++) {
		j = (int)(rand() % n);
		k = (int)(rand() % n);
		tmp = this->getvalue(1, j+1);
		this->setvalue(1, j+1, this->getvalue(1, k+1));
		this->setvalue(1, k+1, tmp);
	}
}


matrix matrix::strip_row(int i)
{
	ASSERT(0 < i && i <= row);
	i -= 1;

	matrix result(1, column);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;

	for (int j = 0; j < column; j++)
		*(ptr_out++) = *(ptr_a + (i * column) + j);

	return result;
}

matrix matrix::strip_column(int i)
{
	ASSERT(0 < i && i <= column);
	i -= 1;

	matrix result(row, 1);

	double* ptr_out	 = result.data;
	double* ptr_a	 = data;

	for (int j = 0; j < row; j++)
		*(ptr_out++) = *(ptr_a + (j * column) + i);

	return result;
}

matrix matrix::row_add()
{
	matrix result(1, row);
	double sum;

	for (int j = 1; j <= row; j++) {
		sum = 0;
		for (int i = 1; i <= column; i++) {
			sum += this->getvalue(j, i);
		}
		result.setvalue(1, j, sum);
	}

	return result;
}

matrix matrix::column_add()
{
	matrix result(1, column);
	double sum;

	for (int j = 1; j <= column; j++) {
		sum = 0;
		for (int i = 1; i <= row; i++) {
			sum += this->getvalue(i, j);
		}
		result.setvalue(1, j, sum / column);
	}

	return result;
}

double matrix::median()
{
	matrix tmp(row, column);
	tmp = *this;
	tmp = tmp.bubblesort();
	double* ptr_1 = tmp.data;
	double* ptr_2 = tmp.data;
	return (length % 2) ? (*(ptr_1) + ((length) / 2 )) : ( (*(ptr_1 + (length / 2 - 1))) + (*(ptr_2 + (length / 2))) ) / 2;
}

matrix matrix::median_column()
{
	matrix result(1, column);
	double median_value;
	for (int j = 1; j <= column; j++) {
		//this->strip_column(j).print("column");

		median_value = this->strip_column(j).median();

		result.setvalue(1, j, median_value);
	}
	return result;	
}

matrix matrix::bubblesort()
{
	double tmp;

	matrix result(row, column);
	result = *this;
	double* arr = result.data;


	for(int i = 0; i < (length - 1); i++)
	{
		for(int j = 0; j < (length - i - 1); j++)
		{
			if(arr[j] > arr[j+1])
			{
				tmp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = tmp;
			}
		}
	}
	return result;
 }

void matrix::print(std::string str)
{
	double* ptr = data;

	#if _DEBUG_MATRIX_
		std::cout << row << "x" << column << " "<< str << " @ " << this << std::endl;
	#else
		std::cout << row << "x" << column << " "<< str << std::endl;
	#endif

	for (int i = 0; i < row; i++) {
		std::cout << " ";
		for (int j = 0; j < column; j++) {
			std::cout.width(12);
			std::cout << std::left << *(ptr++);
		}
		std::cout << std::endl;
	}
}

matrix::~matrix()
{
	#if _DEBUG_MATRIX_
		std::cout << "[#] Destructor called. Memory location: " << this << std::endl;
	#endif

	SAFE_FREE(data);
}


