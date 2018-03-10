import random, time
import opt.ga.Vertex as Vertex
import opt.ga.MaxKColorFitnessFunction as MaxKColorFitnessFunction
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import dist.DiscreteDependencyTree as DiscreteDependencyTree
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
from sys import stdout

N = 150
L = 12
K = 24

data = {'RHC': [0, 0], 'SA': [0, 0], 'GA': [0, 0], 'MIMIC': [0, 0]}

for i in xrange(10):

    stdout.write("\nRunning kColoring iteration %d...\n" % (i + 1))

    random.seed(N * L)
    vertices = []

    for _ in xrange(N):
        vertex = Vertex()
        vertices.append(vertex)
        vertex.setAdjMatrixSize(L)

        for _ in xrange(L):
            vertex.getAadjacencyColorMatrix().add(random.randint(0, N * L - 1))

    ef = MaxKColorFitnessFunction(vertices)
    odd = DiscretePermutationDistribution(K)
    nf = SwapNeighbor()
    mf = SwapMutation()
    cf = SingleCrossOver()

    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

    df = DiscreteDependencyTree(.1)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    start_time = time.time()
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 20000)
    fit.train()
    value = ef.value(rhc.getOptimal())
    stdout.write("RHC %s Value: %d. Time: %0.03f\n" % (ef.foundConflict(),
                                                     value, time.time() - start_time))

    data['RHC'][0] += value
    if ef.foundConflict() == "Found Max-K Color Combination !":
        data['RHC'][1] += 1

    start_time = time.time()
    sa = SimulatedAnnealing(1E12, .1, hcp)
    fit = FixedIterationTrainer(sa, 20000)
    fit.train()
    value = ef.value(sa.getOptimal())
    stdout.write("SA %s Value: %d. Time: %0.03f\n" % (ef.foundConflict(),
                                                        value, time.time() - start_time))

    data['SA'][0] += value
    if ef.foundConflict() == "Found Max-K Color Combination !":
        data['SA'][1] += 1

    start_time = time.time()
    ga = StandardGeneticAlgorithm(300, 150, 50, gap)
    fit = FixedIterationTrainer(ga, 50)
    fit.train()
    value = ef.value(ga.getOptimal())
    stdout.write("GA %s Value: %d. Time: %0.03f\n" % (ef.foundConflict(),
                                                        value, time.time() - start_time))

    data['GA'][0] += value
    if ef.foundConflict() == "Found Max-K Color Combination !":
        data['GA'][1] += 1

    start_time = time.time()
    mimic = MIMIC(300, 100, pop)
    fit = FixedIterationTrainer(mimic, 5)
    fit.train()
    value = ef.value(mimic.getOptimal())
    stdout.write("MIMIC %s Value: %d. Time: %0.03f\n" % (ef.foundConflict(),
                                                        value, time.time() - start_time))

    data['MIMIC'][0] += value
    if ef.foundConflict() == "Found Max-K Color Combination !":
        data['MIMIC'][1] += 1

stdout.write("\nResults:\n")

for key in data:
    stdout.write("%s Average Score: %0.01f, Num. Color Combinations Found: "
                 "%d\n" % (key, data[key][0]/10.0, data[key][1]))






