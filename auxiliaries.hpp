#include "eigenreq.hpp"
#include <fstream>
#include <vector>
#include <iostream>

class InfoPair 
{
	private:
	int i;
	double v;
	
	public:
	InfoPair( int iF, double vF );
	int getI();
	double getV();
	void display();
};

double normaldistribution( double m, double s );

MatrixXd randOrth( int n );

void merge( double *v, double *x, int a, int m, int b );

void mergeSort( double *v, double *x, int a, int b );

void sortVectors( VectorXd &v, VectorXd &x );

void permuteVector( VectorXd &v, VectorXd ix );

void permuteMatrix( MatrixXd &M, VectorXd ix );

VectorXd randPerm( int n );

double median( VectorXd v );

double percentile( VectorXd v, double p );

void xIntoUnitCube( VectorXd x, VectorXd &bx, VectorXd &ix );

VectorXd scale( VectorXd x, VectorXd lb, VectorXd ub );

MatrixXd scaleM( MatrixXd M, VectorXd lb, VectorXd ub );

bool dominates( VectorXd x, VectorXd y );

VectorXd nDomRank( MatrixXd M, bool disp );

VectorXd hyperVolRank2( MatrixXd M, bool disp );
