#include "testfunctions.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int main( int argc, char *argv[] )
{
    // command line arguments: parameterFileName, fitnessFileName
	assert( argc == 3 );
	string parameterFileName = argv[ 1 ];
	string fitnessFileName = argv[ 2 ];

	// simple test case
	// with lower left quadrant of (shifted) unit circle as Pareto front
	string f1Name = "scaledSine";
	string f2Name = "scaledCosine";
	int N = 2;
	int seed = 0;

	string line;
	stringstream ss;
	// read parameter vector from file 
	VectorXd x = VectorXd::Zero( N );
	ifstream parameterFile( parameterFileName.c_str() );
	int i = 0;
	while ( getline( parameterFile, line ) )
	{
//		printf( "%s\n", line.c_str() );
		ss.str( "" ); ss.clear(); ss << line;
		assert( ss >> x( i ) );
		i++;
	}
	parameterFile.close();
	assert( i == N );
//	cout << "x:" << endl << x << endl << endl;

	// calculate both objectives and write them to fitness file
	TTestFunctions testF( N, seed );
	testF.display();
	double f1 = testF.f( f1Name, x );
	double f2 = testF.f( f2Name, x );
	ofstream fitnessFile( fitnessFileName.c_str() );
	fitnessFile << f1 << endl << f2 << endl;
	fitnessFile.close();	
}
