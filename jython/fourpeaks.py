import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array
from sys import stdout, exit
import pickle, os, time



"""
Commandline parameter(s):
   none
"""

iterations_data = {}
iterations_file = "four_peaks_data.pickle"

if os.path.isfile(iterations_file):
    stdout.write("\nFour Peaks Data found.\n")
    exit(0)

N=200
T=N/10
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)



x = xrange(200, 3200, 200)
optimal_value = {'RHC': [], 'SA': [], 'GA': [], 'MIMIC': []}


for item in x:
    stdout.write("\nRunning Four Peaks with %d iterations...\n" % item)

    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, item)
    start = time.time()
    fit.train()
    end = time.time()
    value = ef.value(rhc.getOptimal())
    stdout.write("RHC took %0.03f seconds and found value %d\n" % (end -
                                                                  start, value))
    optimal_value['RHC'].append(value)

    sa = SimulatedAnnealing(1E11, .95, hcp)
    fit = FixedIterationTrainer(sa, item)
    start = time.time()
    fit.train()
    end = time.time()
    value = ef.value(sa.getOptimal())
    stdout.write("SA took %0.03f seconds and found value %d\n" % (end -
                                                                   start, value))
    optimal_value['SA'].append(value)

    ga = StandardGeneticAlgorithm(200, 100, 20, gap)
    fit = FixedIterationTrainer(ga, item)
    start = time.time()
    fit.train()
    end = time.time()
    value = ef.value(ga.getOptimal())
    stdout.write("GA took %0.03f seconds and found value %d\n" % (end -
                                                                   start, value))
    optimal_value['GA'].append(value)

    mimic = MIMIC(200, 20, pop)
    fit = FixedIterationTrainer(mimic, item)
    start = time.time()
    fit.train()
    end = time.time()
    value = ef.value(mimic.getOptimal())
    stdout.write("MIMIC took %0.03f seconds and found value %d\n" % (end -
                                                                   start, value))
    optimal_value['MIMIC'].append(value)

with open(iterations_file, 'wb') as file:
    pickle.dump(optimal_value, file, pickle.HIGHEST_PROTOCOL)
    stdout.write("\nFour Peaks Data saved.\n")
