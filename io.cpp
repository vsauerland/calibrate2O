#include "io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace Eigen; 

void writeScaledVector( VectorXd x, VectorXd lb, VectorXd ub, int nI, int k )
{
	int N = x.size();
	assert( lb.size() == N && ub.size() == N );
	stringstream fileName;
	fileName << "parameters_" << nI + 1 << "_" << k + 1 << ".txt";
	ofstream f( fileName.str().c_str() );
	f.precision( 16 );
	for ( int i = 0; i < N; i++ )
	{
		f << lb( i ) + ( ub( i ) - lb( i ) ) * x( i ) << "\n";
	}
	f.close();
}

VectorXd readScaledVector( VectorXd lb, VectorXd ub, int nI, int k )
{
	assert( ub.size() == lb.size() );
	int N = lb.size();
	VectorXd x = VectorXd::Zero( N );
	stringstream fileName;
	fileName << "parameters_" << nI << "_" << k + 1 << ".txt";
	ifstream f( fileName.str().c_str() );
	string line;
	stringstream ss;
	double value;
	for ( int i = 0; i < N; i++ )
	{
		getline( f, line );
		ss.str( "" ); ss.clear(); ss << line;
		assert ( ss >> value );
		x( i ) = ( value - lb( i ) ) / ( ub( i ) - lb( i ) );
	}
	f.close();
	return( x );
}

void readResult( int nI, int k, double *fitness1, double *fitness2 )
{
	string line;
	stringstream ss;
	stringstream fileName;
	fileName << "fitness_" << nI << "_" << k + 1 << ".txt";
	ifstream file( fileName.str().c_str() );
	getline( file, line );
	if ( line.find( "." ) == string::npos ) *fitness1 = 1e99;
	else
	{
		ss << line;
		assert( ss >> *fitness1 );
	}
	getline( file, line );
	if ( line.find( "." ) == string::npos ) *fitness2 = 1e99;
	else
	{
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> *fitness2 );
	}
	file.close();
}

void writeAlgVars( int nI, int counteval, Population p, MatrixXd X )
{
	stringstream fileName;
	fileName << "algVars_" << nI + 1 << ".txt";
	ofstream f( fileName.str().c_str() );
	f.precision( 16 );
	// 1.) write counteval
	f << "counteval_" << nI << "\n" << counteval << "\n\n";
	// 2.) write population size
	f << "population_" << nI << ":\n\n";
	f << "lambdaMO_" << nI << "\n" << p.getLambdaMO() << "\n\n";
	int lambdaMO = p.getLambdaMO();
	int N = p.getN();
	for ( int k = 0; k < lambdaMO; k++ )
	{
		Individual a = p.getS()[ k ];
		f << "individual_" << nI << "_" << k << ":\n";
		// 2.1.) write cmaes constant cp
		f << "cp_" << nI << "_" << k << "\n" << a.getCp() << "\n";
		// 2.2.) write cmaes constant d
		f << "d_" << nI << "_" << k << "\n" << a.getD() << "\n";
		// 2.3.) write cmaes constant pSuccTarget
		f << "pSuccTarget_" << nI << "_" << k << "\n" << a.getPSuccTarget() << "\n";
		// 2.4.) write cmaes constant pTresh
		f << "pTresh_" << nI << "_" << k << "\n" << a.getPTresh() << "\n";
		// 2.5.) write cmaes constant cc
		f << "cc_" << nI << "_" << k << "\n" << a.getCc() << "\n";
		// 2.6.) write cmaes constant c1
		f << "c1_" << nI << "_" << k << "\n" << a.getC1() << "\n";
		// 2.7.) write dynamic cmaes parameter x
		f << "x_" << nI << "_" << k << "\n";
		VectorXd x = a.getX();
		assert( x.size() == N );
		for ( int i = 0; i < N; i++ )
		{
			f << x( i ) << " ";
		}
		f << "\n";
		// 2.8.) write dynamic cmaes parameter fitness
		f << "fitness_" << nI << "_" << k << "\n";
		f << a.getFitness1() << " " << a.getFitness2() << "\n";
		// 2.9.) write dynamic cmaes parameter pSuccAv
		f << "pSuccAv_" << nI << "_" << k << "\n" << a.getPSuccAv() << "\n";	
		// 2.10.) write dynamic cmaes parameter sigma
		f << "sigma_" << nI << "_" << k << "\n" << a.getSigma() << "\n";	
		// 2.11.) write dynamic cmaes parameter pc
		f << "pc_" << nI << "_" << k << "\n";
		VectorXd pc = a.getPc();
		assert ( pc.size() == N );
		for ( int i = 0; i < N; i++ )
		{
			f << pc( i ) << " ";
		}
		f << "\n";
		// 2.12.) write dynamic cmaes parameter C
		f << "C_" << nI << "_" << k << "\n";
		MatrixXd C = a.getC();
		assert ( C.cols() == N && C.rows() == N );
		for ( int i = 0; i < N; i++ )
		{
			for ( int j = 0; j < N; j++ )
			{
				f << C( i, j ) << " ";
			}
			f << "\n";
		}
		f << "\n";
	} // individual loop
	// 3.) write matrix X of current actual (unit cube) samples
	f << "actual samples (rows) X_" << nI << "\n";
	assert ( X.cols() == N && X.rows() == lambdaMO );
	for ( int k = 0; k < lambdaMO; k++ )
	{
		for ( int i = 0; i < N; i++ )
		{
			f << X( k, i ) << " ";
		}
		f << "\n";
	}
	f.close();
}

void readAlgVars( int nI, int *counteval, Population &p, MatrixXd &X )
{
	int N = p.getN();
	string line;
	stringstream ss;
	stringstream fileName;
	fileName << "algVars_" << nI << ".txt";
	ifstream f( fileName.str().c_str() );
	// 1.) read counteval
	getline( f, line ); getline( f, line );
	ss.str( "" ); ss.clear(); ss << line;
	assert( ss >> *counteval );
//	printf( "readAlgVars:: counteval = %i\n", *counteval );
	// 2.) read population size
	for ( int i = 0; i < 5; i++ ) getline( f, line );
	ss.str( "" ); ss.clear(); ss << line;
	int lambdaMO;
	assert( ss >> lambdaMO );
	assert( lambdaMO == p.getLambdaMO() );
//	printf( "readAlgVars:: lambdaMO = %i\n", lambdaMO );
	for ( int k = 0; k < lambdaMO; k++ )
	{
		// define individual attribute variables to be read
		double cp;
		double d;
		double pSuccTarget;
		double pTresh;
		double cc;
		double c1;
		VectorXd x = VectorXd::Zero( N );
		double fitness1;
		double fitness2;
		double pSuccAv;
		double sigma;
		VectorXd pc = VectorXd::Zero( N );
		MatrixXd C = MatrixXd::Zero( N, N );
		// 2.1.) read cmaes constant pc
		for ( int i = 0; i < 4; i++ ) getline( f, line );
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> cp );
//		printf( "readAlgVars:: cp = %f\n", cp );
		// 2.2.) read cmaes constant d
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> d );
//		printf( "readAlgVars:: d = %f\n", d );
		// 2.3.) read cmaes constant pSuccTarget
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> pSuccTarget );
//		printf( "readAlgVars:: pSuccTarget = %f\n", pSuccTarget );
		// 2.4.) read cmaes constant pTresh
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> pTresh );
//		printf( "readAlgVars:: pTresh = %f\n", pTresh );
		// 2.5.) read cmaes constant cc
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> cc );
//		printf( "readAlgVars:: cc = %f\n", cc );
		// 2.6.) read cmaes constant c1
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> c1 );
//		printf( "readAlgVars:: c1 = %f\n", c1 );
		// 2.7.) read dynamic cmaes parameter x
		getline( f, line ); getline( f, line );
		ss.str( "" ); ss.clear(); ss << line;
		for ( int i = 0; i < N; i++ ) assert( ss >> x( i ) );
//		printf( "readAlgVars:: x:\n" );
//		cout << x << endl << endl;
		// 2.8.) read dynamic cmaes parameter fitness
		getline( f, line ); getline( f, line );
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> fitness1 );
		assert( ss >> fitness2 );
//		printf( "readAlgVars:: fitness2 = %f\n", fitness2 );
		// 2.9.) read dynamic cmaes parameter pSuccAv
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> pSuccAv );
//		printf( "readAlgVars:: pSuccAv = %f\n", pSuccAv );
		// 2.10.) read dynamic cmaes parameter sigma
		getline( f, line ); getline( f, line );	
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> sigma );
//		printf( "readAlgVars:: sigma = %f\n", sigma );
		// 2.11.) read dynamic cmaes parameter pc
		getline( f, line ); getline( f, line );
		ss.str( "" ); ss.clear(); ss << line.c_str();
		for ( int i = 0; i < N; i++ ) assert( ss >> pc( i ) );
//		printf( "readAlgVars:: pc:\n" );
//		cout << pc << endl << endl;
		// 2.12.) read dynamic cmaes parameter C
		getline( f, line );
		for ( int i = 0; i < N; i++ )
		{
			getline( f, line );
			ss.str( "" ); ss.clear(); ss << line.c_str();
			for ( int j = 0; j < N; j++ )
			{
				assert( ss >> C( i, j ) );
			}
		}
//		printf( "readAlgVars:: C:\n" );
//		cout << C << endl << endl;
		Individual a( cp, d, pSuccTarget, pTresh, cc, c1, x, fitness1, fitness2, pSuccAv, sigma, pc, C );
		p.insert( a );
	}
	// 3.) read matrix X of current actual (unit cube) samples
	getline( f, line ); getline( f, line );
	for ( int k = 0; k < lambdaMO; k++ )
	{
		getline( f, line );
		ss.str( "" ); ss.clear(); ss << line.c_str();
		for ( int i = 0; i < N; i++ )
		{
			assert( ss >> X( k, i ) );
		}
	}
//	printf( "readAlgVars:: X:\n" );
//	cout << X << endl << endl;
	f.close();
	assert( p.getS().size() == p.getLambdaMO() );
}
