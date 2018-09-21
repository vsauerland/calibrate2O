#include "eigenreq.hpp"
#include <string>

class TTestFunctions
{
	private:
	MatrixXd O1;
	MatrixXd O2;
	double griewank( VectorXd x );
	double rosenbrock( VectorXd x );
	double sphere( VectorXd x );
	double ssphere( VectorXd x );
	double schwefel( VectorXd x );
	double cigar( VectorXd x );
	double cigtab( VectorXd x );
	double cigtabRotateA( VectorXd x );
	double cigtabRotateB( VectorXd x );
	double cigtabRotateC( VectorXd x );
	double tablet( VectorXd x );
	double elli( VectorXd x );
	double elliRotateA( VectorXd x );
	double elliRotateB( VectorXd x );
	double elliRotateC( VectorXd x );
	double plane( VectorXd x );
	double twoaxes( VectorXd x );
	double parabR( VectorXd x );
	double sharpR( VectorXd x );
	double diffpow( VectorXd x );
	double rastrigin10( VectorXd x );
	double scaledSine( VectorXd x );
	double scaledCosine( VectorXd x );
	
	public:
	TTestFunctions( int n, int seed );
	void display();
	double f( const string &fName, VectorXd x );
};
