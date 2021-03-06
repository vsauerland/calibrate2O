# calibrate2O

The package contains a parallel implementation of the Covariance Matrix
Adaption Evolution Strategy for Multiple Objectives (MO-CMAES) [IHR07] for
Linux platforms.
The current version is supposed to optimize the parameters of models with
respect to two different targets, simultaneously.
It is most suitable if the model run time is approximately the same for
different parameter sets.

Approach
========

Optimization is carried out with regard to two different objectives (model
misfit metrics) by either using a single job "unchained.job" (recommended if
model run times are short, say, less than 10 minutes) or by chain jobs
"serial.job" and "parallel.job" (launched with "serial.job" and recommended
if model run times are long).
Both variants alter MO-CMAES optimization steps with multiple parallel model
simulations.
In each iteration, MO-CMAES samples a population of lambdaMO parameter vectors
from multi-variate normal distributions and writes them to files called
"parameters_i_j.txt", where i is the current iteration number and j is the
session number in {1,...,lambdaMO}.
Subsequently, lambdaMO model simulations are executed in parallel for each of
the stored parameter vectors and write corresponding misfit values to files.
After all model simulations have terminated, a new MO-CMAES optimization step is
started, reads misfit values from files, updates the population of normal
distributions and samples and writes new parameter vectors to files, again.
After iteration i, MO-CMAES writes necessary data for the next iteration (all
dynamic variables of the algorithm) to a file called "algVars_i.txt" (which is
read by MO-CMAES in iteration i+1).
Operational settings and an iteration history are recorded in the interface file
"nIter.txt", which is accessed by MO-CMAES and job files.


Usage
=====

Generating MO-CMAES executables:
--------------------------------

The package consists of the files
* README.md        		// this file
* cmaes_2o.cpp     		// the main source of the MO-CMAES algorithm
* individual.cpp/hpp	// class representing a single (1+1)-CMAES individual
* population.cpp/hpp	// class representing a population of the MO-CMAES
* io.cpp/hpp			// file IO functions required by MO-CMAES
* auxiliaries.cpp/hpp   // auxiliary functions required by MO-CMAES
* eigenreq.hpp      	// required EIGEN C++ includes and namespaces
* testfunctions.cpp/hpp // collection of univariate test objective functions
* testCase.cpp			// a fast two-objective example
* Makefile      		// generates executables
* nIter0.dat        	// example interface file (with operational settings)
* serial.job        	// serial job file (starting MO-CMAES optimizer)
* parallel.job      	// parallel job file (starting parallel model runs)
* unchained.job			// single job file (altering MO-CMAES and model runs)

In order to generate executables, the "Eigen" C++ algebra package must be
available on the system. It can be downloaded from
  http://eigen.tuxfamily.org
as ".tar.gz" file and simply extracted to some place on your system without any
installation process (cf. http://eigen.tuxfamily.org/dox/GettingStarted.html).
The Makefile must be adapted to contain the right path then, i.e., you have to
change the Makefile line spelling
  EIGEN = -I /path/Eigen/
to contain your own path to the Eigen folder.
The MO-CMAES executables should then be generated by typing "make".

Interface to your model:
------------------------

Each model simulation is supposed to read the required parameters from
the corresponding file ("parameters_i_j.txt") and to write
the result (the two objectives) to a corresponding file "fitness_i_j.txt".
The names of both files might be passed, e.g., as command line arguments.
The files are simple text files storing parameter values (objective values)
row by row, i.e. each "parameter_i_j.txt" has n rows
(where n is the number of free parameters)
and each "fitness_i_j.txt" has two rows.

Operational settings are given in the file "nIter.txt" which will be accessed by the jobfiles and also serves as iteration history.
The file "nIter0.dat" is a template for the Slurm job scheduling system that
can/must be copied to "nIter.txt" and adapted to fit your needs.
The settings in the template "nIter0.dat" correspond to a simple test case, where
the Pareto front is the (shifted) lower left quadrant of the unit circle.

Also the file "unchained.job"/"parallel.job" which execute multiple model
simulations in parallel must be modified:
* The header must be adapted with regard to the batch system you use
* The wall-time, the number of nodes and the number of processors per node have
  to be adapted to fit your requirements (according to the requirements for a
  single simulation of your model, multiplied with the number of sessions set
  in "nIter.txt").
* The line that executes "./testCase" has to be adapted to execute your own
  model, i.e., it will look similar to a corresponding model execution in a
  job-file which you use to launch a single model simulation.

Launching and continuing optimization:
--------------------------------------

Using the job scheduling system PBS Pro, an optimization is launched by typing
"msub serial.job" (for the chain job approach) or "msub unchained.job".
Using the Slurm job scheduling system, the corresponding submission would read
"sbatch serial.job" or "sbatch unchained.job".
NOTE, that the headers of the example jobfiles in this package are for Slurm.
It is possible to continue a terminated optimization by increasing the
iteration number in nIter.txt and submitting "serial.job" ("unchained.job"),
again. Also in the case of an interruption caused by technical problems,
optimization can be continued at the point it has been interrupted. In that
case, the last iteration in "nIter.txt" must be deleted if the fitness values of
that iteration have not all been written to the corresponding files!


Literature
==========
 
[IHR07] Igel, C., Hansen, N., Roth, S.,
        *Covariance Matrix Adaptation for Multi-objective Optimization*,
        Evolutionary Computation 15(1):1-28 (2007),
        https://doi.org/10.1162/evco.2007.15.1.1
