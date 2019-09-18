#include "mlp.h"
#include "matrix.h"

int main()
{
	// Cluster
	matrix input(4, 2, ARRAY(0, 0, 
				 0, 1, 
				 1, 0,
				 1, 1));

	matrix output(4, 1, ARRAY(1,
				  0,
				  0,
				  1));

	input.print("Input");
	output.print("Output");

	// Declare Network
	mlp xnor_net(6, 6, input.getwidth(), output.getwidth());
	// Learning phase
	double error = xnor_net.learn(input, output);
	printf("Validation value: %.15f\n", error);

	// Testing phase
	matrix test_1(1, 2, ARRAY(0, 0));
	matrix test_2(1, 2, ARRAY(0, 1));
	matrix test_3(1, 2, ARRAY(1, 0));
	matrix test_4(1, 2, ARRAY(1, 1));
	xnor_net.solve(test_1).print("TEST");
	xnor_net.solve(test_2).print("TEST");
	xnor_net.solve(test_3).print("TEST");
	xnor_net.solve(test_4).print("TEST");

	return 0;
}