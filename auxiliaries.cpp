#include "auxiliaries.hpp"

using namespace Eigen;

InfoPair::InfoPair( int iF, double vF )
{
	i = iF;
	v = vF;
}

int InfoPair::getI()
{
	return( i );
}

double InfoPair::getV()
{
	return( v );
}

void InfoPair::display()
{
	printf( "%i, %f\n", i, v );
}

double normaldistribution( double m, double s )
// generate a normal distributed random number
{
	double r1 = ( double )rand() / RAND_MAX;
	double r2 = ( double )rand() / RAND_MAX;
	double pi = 3.141592653589793;
	return ( s * sqrt( -2 * log( r1 ) ) * sin( 2 * pi * r2 ) + m );
}

MatrixXd randOrth( int n )
// generate a random orthonormal (n x n) matrix
{
	MatrixXd M = MatrixXd::Zero( n, n );
	for ( int i = 0; i < n; i++ ) for ( int j = 0; j < n; j++ )
		M( i, j ) = normaldistribution( 0, 1 );
	MatrixXd O = M.householderQr().householderQ();
	return( O );
}

void merge( double *v, double *x, int a, int m, int b )
// merge two consecutive segments of a vector 
// and arrange a second vector correspondingly
// 
// v, x     vectors
// a, m, b  component indizes ( counting from 0 )
//
// both subvectors of consecutive components from a up to m 
// and from m+1 up to b of v are supposed to be in ascending order 
// merging brings all components from a up to b in ascending order
// and arranges the components of x correspondingly  
{
	double	*w, *y;
	int	i, j, k;
	w = ( double* ) calloc( m + 1 - a, sizeof( double ) );
	y = ( double* ) calloc( m + 1 - a, sizeof( double ) );
	// copy left segments of v and x to auxially vectors w and y:
	for( i = 0; i < m + 1 - a; i++ )
	{
		*( w + i ) = *( v + a + i );
		*( y + i ) = *( x + a + i );
	}
	i = 0;
	j = m + 1;
	k = a;
	// copy back the next greater elements:
	while( k < j && j <= b )
	{
		if( *( w + i ) <= *( v + j ) )
		{
			*( v + k ) = *( w + i );
			*( x + k ) = *( y + i );
			k++;
			i++;
		} else
		{
			*( v + k ) = *( v + j );
			*( x + k ) = *( x + j );
			k++;
			j++;	
		} 
	}
	// copy back remaining elements of w:
	while ( k < j )
	{
		*( v + k ) = *( w + i );
		*( x + k ) = *( y + i );
		k++;
		i++;
	}
	free( w );
	free( y );
}

void mergeSort( double *v, double *x, int a, int b )
// sort one double array and permute a second double array accordingly
//
// v: array to be sorted in ascending order
// x: array to be permuted like v
// a,b: components a to b (0 <= a <= b <= length(v) - 1 ) are sorted
{
	int m;
	if( a < b )
	{
		m = ( a + b ) / 2;
		mergeSort( v, x, a, m );
		mergeSort( v, x, m + 1, b );
		merge( v, x, a, m, b );
	}	
}

void sortVectors( VectorXd &v, VectorXd &x )
// sort a vector and permute a second vector accordingly
{
	int n = v.size();
	// copy vectors to some double arrays
	double *va, *xa;
	va = ( double* )calloc( n, sizeof( double ) );
	xa = ( double* )calloc( n, sizeof( double ) );
	for ( int i = 0; i < n; i++ )
	{
		va[ i ] = v( i );
		xa[ i ] = x( i );
	}
	// sort double arrays
	mergeSort( va, xa, 0, n - 1 );
	// copy back double arrays to vectors
	for ( int i = 0; i < n; i++ )
	{
		v( i ) = va[ i ];
		x( i ) = xa[ i ];
	}
	free( va );
	free( xa );
}

void permuteVector( VectorXd &v, VectorXd ix )
// permute components of a vector
// v  vector that is permuted (length n)
// ix vector with n different integer components in {0,1,...,n-1}
//    ix represents a permutation w.r.t. which v is permuted
{
	int n = v.size();
	VectorXd w = VectorXd::Zero( n );
	for ( int i = 0; i < n; i++ ) w( ix( i ) - 1 ) = v( i );
	for ( int i = 0; i < n; i++ ) v( i ) = w( i );
}

void permuteMatrix( MatrixXd &M, VectorXd ix )
// permute rows of a matrix
// M  matrix with n rows that are permuted
// ix vector with n different integer components in {0,1,...,n-1}
//    ix represents a permutation w.r.t. which the rows of M are permuted
{
	int n = M.rows();
	int m = M.cols();
	MatrixXd M2 = MatrixXd::Zero( n, m );
	for ( int i = 0; i < n; i++ ) M2.row( ix( i ) - 1 ) = M.row( i );
	for ( int i = 0; i < n; i++ ) M.row( i ) = M2.row( i );
}

VectorXd randPerm( int n )
// generate a random permutation represented as a vector
{
	VectorXd result = VectorXd::Zero( n );
	for ( int i = 0; i < n; i++ )
	{
		int j = rand() % n;
		while ( result( j ) > 0 ) j = rand() % n;
		result( j ) = i + 1;
	}
	for ( int i = 0; i < n; i++ ) result( i ) = result( i ) - 1;
	return ( result );
}

double median( VectorXd v ) // Computes median of a vector v
{
	int n = v.size();
	VectorXd dummy( n );
	sortVectors( v, dummy );
	if ( n % 2 == 1 ) return( v( n / 2 ) );
	else return( 0.5 * ( v( n / 2 - 1 ) + v( n / 2 ) ) );
}

double percentile( VectorXd v, double p ) // Computes the percentile p from vector v
{
	int n = v.size();
	double r;	
	VectorXd dummy( n );
	sortVectors( v, dummy );
	if ( p <= 100 * ( 0.5 / n ) ) r = v( 0 );
	else if ( p >= 100 * ( ( n - 0.5 ) / n ) ) r = v( n - 1 );
	else
	{
		// find largest index smaller than required percentile
		VectorXd indices = VectorXd::LinSpaced( n, 1, n );
		VectorXd a = 100 * ( indices.array() - 0.5 ) / n;
		int i = ( a.array() >= p ).select( 0, indices ).lpNorm<Eigen::Infinity>() - 1;
		// interpolate linearly
		r = v( i ) + ( v( i + 1 ) - v( i ) ) * ( p - a( i ) ) / ( a( i + 1 ) - a( i ) );
	}
	return( r );
}

void xIntoUnitCube( VectorXd x, VectorXd &bx, VectorXd &ix )
// map a vector x to unit cube (maping outside components onto closest bound)
// bx is the result, ix indicates the out of bounds components of x  
{
	ix = VectorXd::Zero( ix.size() );
	VectorXd ix2 = VectorXd::Zero( ix.size() );
	ix = ( x.array() < 0 ).select( 1, ix );
	bx = ( x.array() < 0 ).select( 0, x );
	ix2 = ( bx.array() > 1 ).select( 1, ix2 );
	bx = ( bx.array() > 1 ).select( 1, bx );
	ix = ix2 - ix;
}

VectorXd scale( VectorXd x, VectorXd lb, VectorXd ub )
// scale a vector x from unit cube
// to rectangle described by lower bounds lb and upper bounds ub
{
	return( lb.array() + ( ub.array() - lb.array() ) * x.array() );
}

MatrixXd scaleM( MatrixXd M, VectorXd lb, VectorXd ub )
// scale matrix M of column vectors from unit cube
// to rectanlge descibed by lower bounds lb and upper bounds ub
{
	int N = M.rows();
	int lambda = M.cols();
	MatrixXd M2( N, lambda );
	for ( int k = 0; k < lambda; k++ )
	{
		M2.col( k ) = scale( M.col( k ), lb, ub );
	}
	return( M2 );
}

bool dominates( VectorXd x, VectorXd y )
// indicates if x < y where "<" is the component wise comparison (cf. 3. in [1])
{
	int n = x.size();
	bool result = false;
	if ( ( x.array() <= y.array() ).cast< int >().sum() == n
		&& ( x.array() < y.array() ).cast< int >().sum() > 0 )
	{
		result = true;
	}
	return( result );
}

VectorXd nDomRank( MatrixXd M, bool disp )
// calculate partial ranking w.r.t. undominated sorting (cf. 3.2.1. in [1])
// implementation effort is O(mn^3), an O(mn^2) implementation is given in [2]
// 
// INPUT:
// M    matrix with entries that are supposed to represent
//      m objective function values of n vectors (individuals): M_i_j = f_j(x_i)
// disp indicates, if details of the algorithm steps are displayed
//
// OUTPUT:
// a vector containing the ranks of the n rows of M (the n individuals)
// w.r.t. undominated sorting (cf. 3.2.1. in [1])
{
	int n = M.rows();
	int m = M.cols();
	VectorXd v = VectorXd::Zero( n );
	VectorXd oldV = v;
	int currentRank = 1;
	while ( currentRank <= n )
	{
		int count = 0;
		for ( int i = 0; i < n; i++ ) if ( oldV( i ) == 0 )
		{
			bool isDominated = false;
			for ( int j = 0; j < n; j++ ) if ( oldV( j ) == 0 )
			{
				if ( dominates( M.row( j ), M.row( i ) ) )
				{
					isDominated = true;
					if ( disp ) printf( "%i is dominated by %i\n", i, j );
				}
			}
			if ( !isDominated )
			{
				if ( disp ) printf( "since %i is not dominated by unranked, it gets rank %i, now\n", i, currentRank );
				v( i ) = currentRank;
				count++;
			}
		}
		oldV = v;
		if ( disp ) printf( "since %i ones have been ranked with %i, the rank is increased to %i, now\n", count, currentRank, currentRank + count );
		currentRank = currentRank + count;
	}
	return( v );
}

VectorXd hyperVolRank2( MatrixXd M, bool disp )
// calculate ranking w.r.t. hyper volume contributions (cf. 3.2.3. in [1])
// 
// INPUT:
// M    matrix with entries that are supposed to represent
//      2 objective function values of n vectors (individuals): M_i_j = f_j(x_i)
// disp indicates, if details of the algorithm steps are displayed
//
// OUTPUT:
// a vector containg the total ranking of the n individuals using their
// hyper volume contributions as second criterium (cf. 3.2.3. in [1])
//
// using std vectors, this implementation needs quadratic time while Lemma 1
// in [1] proofs sublinear time with suitable data structures (AVL trees)
{
	int n = M.rows();
	int m = M.cols();
	assert( m == 2 );
	VectorXd result = VectorXd::Zero( n );
	// determine partial rank vector nDR w.r.t. non-dominance
	// (defining a partition of the n individuals into subsets within which
	//  crowding distance ranking will be done locally)
	VectorXd nDR = nDomRank( M, disp );
	// determine number of partial ranks (number of Pareto fronts) nP
	// and the vector of corresponding rank values vP
	int nP = 0;
	VectorXi isRank = VectorXi::Zero( n );
	for ( int r = 1; r <= n; r++ )
	{
		int countR = 0;
		for ( int j = 0; j < n; j++ ) if ( nDR( j ) == r ) countR++;
		if ( countR > 0 )
		{
			isRank( r - 1 ) = 1;
			nP++;
		}
	}
	if ( disp ) printf( "there are %i ranks\n", nP );
	VectorXi vP = VectorXi::Zero( nP );
	int i = 0;
	for ( int r = 1; r <= n; r++ ) if ( isRank( r - 1 ) )
	{
		vP( i ) = r;
		i++;
	}
	assert( i == nP );
	if( disp ) cout << "the ranks are:" << endl << vP << endl << endl;
	// calculate the exact rank for each individual in each Pareto front:
	for ( int k = 0; k < nP; k++ )
	{
		// select MSub that corresponds to the current Pareto front
		int nK = 0;
		for ( int i = 0; i < n; i++ ) if ( nDR( i ) == vP( k ) ) nK++;
		if( disp ) printf( "pareto front %i has %i members\n", k + 1, nK );
		VectorXi rankLocal = VectorXi::Zero( nK );
		VectorXi index = VectorXi::Zero( nK );
		MatrixXd MSub = MatrixXd( nK, m );
		int j = 0;
		for ( int i = 0; i < n; i++ ) if ( nDR( i ) == vP( k ) )
		{
			index( j ) = i;
			MSub.row( j ) = M.row( i );
			j++;
		}
		assert( j == nK );
		// matrix A will be optionally updated and displayed for verification
		MatrixXd A = MatrixXd::Zero( nK, 4 );
		A.col( 0 ) = VectorXd::LinSpaced( nK, 0, nK - 1 );
		A.col( 1 ) = index.cast<double>();
		A.col( 2 ) = MSub.col( 0 );
		A.col( 3 ) = MSub.col( 1 );
		if ( disp ) cout << "its this row numbers:\n" << index << endl;
		if ( disp ) cout << "corresponding MSub:\n" << MSub << endl << endl;
		if ( nK > 1 )
		{
			vector<InfoPair> F;
			vector<InfoPair> S;
			// initialize F with row indices and first column of MSub,
			// incresingly sorted by objective values
			VectorXd val = MSub.col( 0 );
			VectorXd ind = VectorXd::LinSpaced( nK, 0, nK - 1 );
			sortVectors( val, ind );
			for ( int i = 0; i < nK; i++ )
			{
				InfoPair a( int( ind( i ) ), val( i ) );
				F.push_back( a );
			}
			assert( F.size() == nK );
			if ( disp )
			{
				printf( "initial F:\n" );
				for ( int ii = 0; ii < F.size(); ii++ ) F[ ii ].display();
				printf( "\n" );
			}
			// initialize S with row indices of MSub and the contributing
			// hypervolumes w.r.t. all rows of MSub, i.e. all nK individuals
			double f1Max = MSub.col( 0 ).maxCoeff();
			double f1Min = MSub.col( 0 ).minCoeff();
			double f2Max = MSub.col( 1 ).maxCoeff();
			double f2Min = MSub.col( 1 ).minCoeff();
			double deltaF1 = max( 1.0, ( double )( f1Max - f1Min ) );
			double deltaF2 = max( 1.0, ( double )( f2Max - f2Min ) );
			double hVmax = 2 * deltaF1 * deltaF2;
			if ( disp ) printf( "f1Max = %f, f1Min = %f, f2Max = %f, f2Min = %f\n", f1Max, f1Min, f2Max, f2Min );
			if ( disp ) printf( "the reference poit hypervolume is %f\n", hVmax );
			// calculate the initial contributing hypervolumes
			val = VectorXd::Zero( nK );
			for ( int i = 0; i < nK; i++ )
			{
				if ( i == F[ nK - 1 ].getI() || i == F[ 0 ].getI() )
				{
					if ( disp ) printf( "%i is boundary element\n", i );
					val( i ) = hVmax;
				}
				else
				{
					int j = 1;
					while ( j < nK - 2 && F[ j ].getI() != i  ) j++;
					assert( F[ j ].getI() == i );
					int i1 = F[ j - 1 ].getI();
					int i2 = F[ j + 1 ].getI();
					val( i ) =   ( MSub( i2, 0 ) - MSub( i, 0 ) )
					           * ( MSub( i1, 1 ) - MSub( i, 1 ) );
					if ( disp ) printf( "contr. hypervolume of %i is %f\n", i, val( i ) );
				}
			} // contr. hypervolumes initialization loop
			ind = VectorXd::LinSpaced( nK, 0, nK - 1 );
			sortVectors( val, ind );
			for ( int i = 0; i < nK; i++ )
			{
				InfoPair a( int( ind( i ) ), val( i ) );
				S.push_back( a );
			}
			assert( S.size() == nK );
			if ( disp )
			{
				printf( "\ninitial S:\n" );
				for ( int ii = 0; ii < S.size(); ii++ ) S[ ii ].display();
				printf( "\n" );
			}
			// now, calculate local ranks in the current Pareto front
			// w.r.t. hyper volume contributions (see [1], Lemma 1)
			// note, that the worst local rank there is 0 and the best
			// local rank is nK-1, but we assign low values to good ranks
			// in accordance with the global ranking
			// first assign ranks of best (boundary) pareto front elements
			rankLocal( S[ nK - 1 ].getI() ) = 1;
			rankLocal( S[ nK - 2 ].getI() ) = 0;
			if ( disp )
			{
				printf( "%i gets rank %i\n", S[ nK - 1 ].getI(), 1 );
				printf( "%i gets rank %i\n", S[ nK - 2 ].getI(), 0 );
			}
			// assign ranks of remaining pareto front elements
			int l = nK - 1; // current remaining rank (worst one)
			if ( nK > 2 ) for ( int t = 0; t < nK - 2; t++ )
			{
				// take (Pareto front index of) worst element
				int i = S[ 0 ].getI();
				// delete current element from S 
				S.erase( S.begin() );
				// assign and update current rank
				rankLocal( i ) = l;
				if ( disp ) printf( "%i gets rank %i\n", i, rankLocal( i ) );
				l--;
				// obtain element i position j in F
				int j = 0;
				while ( F[ j ].getI() != i && j < F.size() ) j++;
				assert( F[ j ].getI() == i );
				// obtain (Pareto front indices of) F-neighbors
				int i1 = F[ j - 1 ].getI();
				int i2 = F[ j + 1 ].getI();
				if ( disp ) printf( "the F-neighbors of %i are %i and %i\n", i, i1, i2 );
				// delete current element i from F
				F.erase( F.begin() + j );
				if ( disp )
				{
					printf( "deleted %i from F and S\n", i );
					printf( "current F:\n" );
					for ( int ii = 0; ii < F.size(); ii++ ) F[ ii ].display();
					printf( "\n" );
					printf( "current S:\n" );
					for ( int ii = 0; ii < S.size(); ii++ ) S[ ii ].display();
					printf( "\n" );
				}
				// delete F-neighbors from S
				int j1 = 0;
				while ( S[ j1 ].getI() != i1 && j1 < S.size() ) j1++;
				assert( S[ j1 ].getI() == i1 );
				S.erase( S.begin() + j1 );
				int j2 = 0;
				while ( S[ j2 ].getI() != i2 && j2 < S.size() ) j2++;
				assert( S[ j2 ].getI() == i2 );
				S.erase( S.begin() + j2 );
				if ( disp )
				{
					printf( "deleted F-neighbors %i and %i of %i from S\n", i1, i2, i );
					printf( "current S:\n" );
					for ( int ii = 0; ii < S.size(); ii++ ) S[ ii ].display();
					printf( "\n" );
					printf( "here is the remaining rows (and there M-numbers) of MSub, again:\n" );
					for ( int ii = 0; ii < F.size(); ii++ )
					{
						cout << A.row( F[ ii ].getI() ) << endl;
					}
					cout << endl;
				}
				// calculate new contributing hyper volumes
				// of both F-neighbors
				double hV1, hV2;
				if ( i1 == F[ F.size() - 1 ].getI() || i1 == F[ 0 ].getI() )
				{
					hV1 = hVmax;
				}
				else
				{
					j1 = 1;
					while ( F[ j1 ].getI() != i1 && j1 < F.size() - 2 ) j1++;
					assert( F[ j1 ].getI() == i1 );
					int i11 = F[ j1 - 1 ].getI();
					int i12 = F[ j1 + 1 ].getI();
					hV1 =   ( MSub( i12, 0 ) - MSub( i1, 0 ) )
					      * ( MSub( i11, 1 ) - MSub( i1, 1 ) );
				}
				if ( disp ) printf( "new hypervolume contibution of %i without %i is %f\n", i1, i, hV1 );
				if ( i2 == F[ F.size() - 1 ].getI() || i2 == F[ 0 ].getI() )
				{
					hV2 = hVmax;
				}
				else
				{
					j2 = 1;
					while ( F[ j2 ].getI() != i2 && j2 < F.size() - 2 ) j2++;
					assert( F[ j2 ].getI() == i2 );
					int i21 = F[ j2 - 1 ].getI();
					int i22 = F[ j2 + 1 ].getI();
					hV2 =   ( MSub( i22, 0 ) - MSub( i2, 0 ) )
					      * ( MSub( i21, 1 ) - MSub( i2, 1 ) );
				}
				if ( disp ) printf( "new hypervolume contibution of %i without %i is %f\n", i2, i, hV2 );
				// reinsert F-neighbors into S
				j1 = 0;
				while ( j1 < S.size() && S[ j1 ].getV() < hV1 ) j1++;
				S.insert( S.begin() + j1, InfoPair( i1, hV1 ) );
				j2 = 0;
				while ( j2 < S.size() && S[ j2 ].getV() < hV2 ) j2++;
				S.insert( S.begin() + j2, InfoPair( i2, hV2 ) );
				if ( disp )
				{
					printf( "reinserted F-neighbors %i and %i of %i into S\n", i1, i2, i );
					printf( "current S:\n" );
					for ( int ii = 0; ii < S.size(); ii++ ) S[ ii ].display();
					printf( "\n" );
				}
			} // ranking loop, if pareto front has more than 2 elements
		} // if pareto front has more than 1 element
		// now, assign the ranks w.r.t. M
		for ( int i = 0; i < nK; i++ )
		{
			result( index( i ) ) = rankLocal( i ) + vP( k );
		}
		if ( disp )
		{
			printf( "so far, the ranks of M are:\n" );
			cout << result << endl << endl;
		}
	} // pareto fronts loop
	return( result );
}



/* -------------------------------- Literature --------------------------------

[1] C. Igel, N. Hansen, and S. Roth (2007). Covariance Matrix Adaption for
    Multi-objective Optimization. Evolutionary Computation 15(1), 28 pages.

[2] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan (2002).
    A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. 
    IEEE Transactions on Evolutionary Computation 6(2), pages 182-197.

-----------------------------------------------------------------------------*/
