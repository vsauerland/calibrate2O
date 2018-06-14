# calibrate2O

The package contains a parallel implementation of the Covariance Matrix
Adaption Evolution Strategy for Multiple Objectives (MO-CMAES) [IHR07] for
Linux platforms.

PURPOSE
-------

The current version is supposed to optimize the parameters of models with
respect to two different targets, simultaneously.
It is most suitable if the model run time is approximately the same for
different parameter sets.

APPROACH
--------

Optimization is carried out with regard to two different objectives (model
misfit metrics) by either using a single job "unchained.job" (recommended if
model run times are short, say, less than 10 minutes) or by alternating chain
jobs "serial.job" and "parallel.job" (launched by "serial.job" and recommended
if model run times are long).
Both variants alter MO-CMAES optimization steps with multiple parallel model
simulations. The number lambdaMO of model simulations corresponds to the number 
of sampled parameter vectors in each MO-CMAES iteration (population size).
In each iteration, MO-CMAES samples lambdaMO parameter vectors from
multi-variate normal distributions and writes them to files called
"parameters_i_j.txt", where i is the current iteration number and j is the
session number in {1,...,lambdaMO}.
NOTE: Each model simulation is supposed to read a parameter file and to write
the result (the two objectives) to a corresponding file "fitness_i_j.txt".
Both files are simple text files, storing parameter values (objective values)
row by row, i.e. "parameter_i_j.txt" has n rows (where n is the number of free
parameters) and "fitness_i_j.txt" has two rows.
After termination of the model simulations, a new MO-CMAES optimization step is
started, reads the fitness values from files, updates the population of normal
distributions and samples and writes new parameter vectors to files, again.
After iteration i, MO-CMAES writes necessary data for the next iteration (all
dynamic variables of the algorithm) to a file called "algVars_i.txt" (which is
read by MO-CMAES in iteration i+1).
Operational settings and an iteration history are recorded in the interface file
"nIter.txt", which is accessed by MO-CMAES and job files.


USAGE
-----

Generating Executables:
-----------------------

The package consists of the files
* README        		// this file
* cmaes_2o.cpp     		// the main source of the MO-CMAES algorithm
* auxiliaries.cpp/hpp   // some auxiliary functions required by MO-CMAES
* eigenreq.hpp      	// path to required EIGEN algebra package and some defs
* testfunctions.cpp 	// collection of testfunctions for optimization
* Makefile      		// generates executables
* nIter0.dat        	// sample interface file (with operational settings)
* serial.job        	// serial job file (starting MO-CMAES optimizer)
* parallel.job      	// parallel job file (starting parallel model runs)
* unchained.job			// single job file (alterning MO-CMAES and model runs)

In order to generate executables, the "Eigen" algebra package must be available
on the system. It can be downloaded from
  http://eigen.tuxfamily.org
as ".tar.gz" file and simply extracted to some place on your system without any
installation process (cf. http://eigen.tuxfamily.org/dox/GettingStarted.html).
The Makefile must be adapted to contain the right path then, i.e., you have to
change the Makefile line spelling
  EIGEN = -I /path/eigen-eigen-36fd1ba04c12/Eigen/
to contain your own path to the Eigen folder.
The MO-CMAES executables should then be generated by typing "make".

Operational settings:
---------------------

Operational settings are done in the file "nIter.txt" which will be accessed by 
the jobfiles and also serves as iteration history.
The file "nIter0.dat" is a template for the PBS Pro job scheduling system that
can/must be copied to "nIter.txt" and adapted to fit your needs.
The settings in the template "nIter0.dat" correspond to a simple testcase, where
the Pareto front is the (shifted) lower left quadrant of the unit circle.

Also the file "unchained.job"/"parallel.job" which run multiple model
simulations in parallel must be modified:
* The walltime, the number of nodes and the number of processors per node have
  to be adapted to fit your requirements (according to the requirements for a
  single simulation of your model, multiplied with the number of sessions set
  in "nIter.txt").
* The line "aprun ..." has to be adapted to execute your own model, i.e., it
  will look similar to a corresponding model execution in a jobfile which you
  use to launch a single model simulation.

Launching Optimization:
---------------------

Using the job scheduling system PBS Pro, an optimization can be launched typing
"msub serial.job" (for the chain job approach) or "msub unchained.job".
Using the Slurm job scheduling system, the corresponding submission would read
"sbatch serial.job" or "sbatch unchained.job".
It is possible to continue a terminated optimization by increasing the
iteration number in nIter.txt and submitting "serial.job" ("unchained.job"),
again. Also in the case of an iterruption caused by technical problems,
optimization can be continued at the point it has been interrupted. In that
case, the last iteration in "nIter.txt" must be deleted if the fitness values of
that iteration have not all been written to the corresponding files!


LITERATURE
----------
 
[IHR07] Igel, C., Hansen, N., Roth, S.,
        *Covariance Matrix Adaptation for Multi-objective Optimization*,
        Evolutionary Computation 15(1):1-28 (2007), DOI:10.1162/evco.2007.15.1.1
