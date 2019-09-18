/*
 _____ __    _____ 
|     |  |  |  _  |
| | | |  |__|   __|
|_|_|_|_____|__|    C++11 Standard
 
Multilayer perceptron (MLP) with backpropagation
A very simple implementation of artificial neural network

Written by Emilio Schinina' (emilioschi@gmail.com), JAN 2019

File:	utils.h
*/


#ifndef _UTILS_H_
#define _UTILS_H_

#define randseed() srand(time(NULL));

#define ASSERT(condition)\
 do { if(!(condition)){ std::cerr << "[ASSERT] "<< "\x1b[31m" << #condition <<  "\x1b[0m" << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; exit(1);}} while(0)

#define	SAFE_FREE(X)\
	do { if((X) != NULL) {free(X); X=NULL;} } while(0)

#endif /* _UTILS_H_ */
