// Implementation of the Covariance Matrix Adaption Evolution Strategy (CMAES)
// This implements the lambdaMOx(1+1) 2-obj. strategy with box constraints
// and two-objective optimization
// We use C++ with the Eigen package for matrix calculations.

#include <iostream>
#include <fstream>
#include <sstream>
#include "testfunctions.hpp"
#include "io.hpp"
#include "auxiliaries.hpp"


int main( int argc, char* argv[] )
{
// ========================= Set/Read Problem Instance ========================
	int N;
	int lambdaMO;
	int seed;
	int nIMax;
	string fName1;
	string fName2;
	int nI;
	int counteval;
	if ( argc < 2 ) 
	{
		printf( "The program requires two coomane line arguments,\n" );
		printf( "current iteration (integer) and instance filename (string).\n" );
		printf( "The associated file is supposed to contain:\n" );
		printf( "- a headline (line 1)\n" );
		printf( "- name of first fitness function (line 2, not used),\n" );
		printf( "- name of second fitness function (line 3, not used),\n" );
		printf( "- problem dimension n (line 4),\n" );
		printf( "- cmaes population size (line 5),\n" );
		printf( "- random integer seed for cmaes (line 6)\n" );
		printf( "- maximum number of iterations (line 7)\n" );
		printf( "- a sub-headline (line 8)\n" );
		printf( "- lower and upper parameter bounds (lines 9 to n + 8)\n\n" );
		exit( 1 );
	}
	// read instance from provided file
	nI = atoi( argv[ 1 ] );
	string inFileName = argv[ 2 ];
	ifstream inFile( inFileName.c_str() );
	string line; stringstream ss;
	getline( inFile, line );
	getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
	assert ( ss >> fName1 );
	getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
	assert ( ss >> fName2 );
	getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
	assert ( ss >> N );
	getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
	assert ( ss >> lambdaMO );
	getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
	assert ( ss >> nIMax );
	getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
	assert ( ss >> seed );
	VectorXd lb = VectorXd::Zero( N );
	VectorXd ub = VectorXd::Zero( N );
	getline( inFile, line );
	for ( int i = 0; i < N; i++ )
	{
		getline( inFile, line ); ss.str( "" ); ss.clear(); ss << line;
		assert( ss  >> lb( i ) >> ub( i ) );
	}
	inFile.close();
	Population p( lambdaMO, N );
	MatrixXd X = MatrixXd( lambdaMO, N ); // matrix of samples

	if ( nI == 0 )
	{
		// generate initial population without evaluated mean values
		printf( "this is CMAES in iteration 0\n" );
		// insert random individuals with standard operational parameters
		double d = 1 + ( double )N / 2;
		double pSuccTarget = 0.181818;
		double cp = pSuccTarget / ( 2 + pSuccTarget );
		double cc = ( double )2 / ( N + 2 ); // time constant for cumulation for C
		double c1 = ( double )2 / ( N * N + 6 ); // learning rate for rank-one update of C
		double pTresh = 0.44;
		double sigma = 0.25; // coordinate wise standard deviation (step-size)
		VectorXd pc = VectorXd::Zero( N ); // evolution path for C
		double pSuccAv = 0;
		MatrixXd C = MatrixXd::Identity( N, N );  // covariance matrix
		MatrixXd X = MatrixXd::Zero( lambdaMO, N ); // matrix of samples
		srand( seed );
		printf( "calling srand( %i ) before making initial random means\n", seed );
		for ( int k = 0; k < lambdaMO; k++ )
		{
			VectorXd x = VectorXd::Random( N );
			x = ( x.array() + 1 ) / 2;
//			X.row( k ) = x.transpose();
			writeScaledVector( x, lb, ub, nI, k );
			Individual a( cp, d, pSuccTarget, pTresh, cc, c1, x, 1e99, 1e99, pSuccAv, sigma, pc, C );
			p.insert( a );
		}
		p.display( false );
		writeAlgVars( 0, lambdaMO, p, X ); // store initial population p with infinity objective values and a matrix X of its means
	}
	else if ( nI == 1 )
	{
		// read and set fitness values of the initial population
		// and draw and write first samples
		readAlgVars( nI, &counteval, p, X );
		for ( int k = 0; k < lambdaMO; k++ )
		{
			double f1, f2;
			readResult( nI, k, &f1, &f2 ); // fitness of v
			printf( "setting fitness of population member %i\n", k );
			p.setF( k, f1, f2 );
			printf( "done\n" );
		}
		printf( "calling srand( %i ) before first sampling in iteration %i\n", seed + counteval, nI );
		srand( seed + counteval );
		for ( int k = 0; k < lambdaMO; k++ )
		{
			VectorXd x = p.getS()[ k ].sample();
			assert( x.size() == N );
			X.row( k ) = x.transpose();
			VectorXd v( N );
			VectorXd dummy( N );
			xIntoUnitCube( x, v, dummy );
			// v is closest feasible unit cube point to x
			writeScaledVector( v, lb, ub, nI, k );
		}
		cout << "and this is the matrix of samples:" << endl << X << endl << endl;
		writeAlgVars( nI, counteval + lambdaMO, p, X ); // store initial population p with actual objective values and a matrix X of its samples
	}
	else // ( nI > 1 )
	{
		// read actual former samples and their fitness values, update population,
		// and draw and write new samples from the new population
		readAlgVars( nI, &counteval, p, X );
		vector<Individual> S = p.getS();
		assert( S.size() == lambdaMO );
		for ( int k = 0; k < lambdaMO; k++ )
		{
			VectorXd x = X.row( k ).transpose(); // x is former k-th sample
			VectorXd v = readScaledVector( lb, ub, nI, k );
			// v is the closest feasible unit cube point to x
			double f1, f2;
			readResult( nI, k, &f1, &f2 ); // fitness of v
			// fitness(x)=fitness(v)+penalty:
			f1 = f1 + 1e-6 * ( v - x ).squaredNorm();
			f2 = f2 + 1e-6 * ( v - x ).squaredNorm();
			S[ k ].setX( x );
			S[ k ].setF( f1, f2 );
		}
		// for update count the number of dominating offspring
		VectorXd succ = VectorXd::Zero( lambdaMO );
		for ( int k = 0; k < lambdaMO; k++ )
		{
			VectorXd FOld = p.getS()[ k ].getF();
			VectorXd FNew = S[ k ].getF();
			if ( dominates( FNew, FOld ) ) succ( k ) = 1;
		}
		// update stepsizes and covariance matrices	
		for ( int k = 0; k < lambdaMO; k++ )
		{
			p.updateStepSize( k, succ( k ) );
			S[ k ].updateStepSize( succ( k ) );
			double sig = S[ k ].getSigma();
			VectorXd deltaX = S[ k ].getX() - p.getS()[ k ].getX();
			S[ k ].updateCovariance( 1 / sig * deltaX );
		}
		// update population
		for ( int k = 0; k < lambdaMO; k++ ) p.insert( S[ k ] );
		assert( p.getS().size() == 2 * lambdaMO );
		p.trim();
		// draw new samples from new population and write to parameter files
		srand( seed + counteval );
		printf( "calling srand( %i ) before sampling in iteration %i\n", seed + counteval, nI );
		for ( int k = 0; k < lambdaMO; k++ )
		{
			VectorXd x = p.getS()[ k ].sample();
			assert( x.size() == N );
			X.row( k ) = x.transpose();
			VectorXd v( N );
			VectorXd dummy( N );
			xIntoUnitCube( x, v, dummy );
			// v is closest feasible unit cube point to x
			writeScaledVector( v, lb, ub, nI, k );
		}
		cout << "and this is the matrix of samples:" << endl << X << endl << endl;
		writeAlgVars( nI, counteval + lambdaMO, p, X ); // store new population p obtained by using former populations samples and new samples X
	} // else ( nI > 1 )
	// update nIter.txt
	double f1Min, f1Max, f2Min, f2Max;
	p.extremes( &f1Min, &f1Max, &f2Min, &f2Max );
	ofstream file;
	file.open( "nIter.txt", ofstream::out | ofstream::app );
	file << nI + 1 << " [ " << f1Min << ", " << f1Max << " ]";
	file << " [ " << f2Min << ", " << f2Max << " ]\n";
	file.close();
}

/* -------------------------------- Literature --------------------------------

[1] C. Igel, N. Hansen, and S. Roth (2007). Covariance Matrix Adaption for
    Multi-objective Optimization. Evolutionary Computation 15(1). 28 pages.

[2] N. Hansen, A.S.P. Niederberger, L. Guzzella and P. Koumoutsakos (2009).
    A Method for Handling Uncertainty in Evolutionary Optimization with an
    Application to Feedback Control of Combustion.
    IEEE Transactions on Evolutionary Computation, 13(1), pp. 180-197

[3] CMAES website: https://www.lri.fr/~hansen/cmaesintro.html

-----------------------------------------------------------------------------*/
