"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying the pima
dataset.

Based on abalone_test.py
"""
from __future__ import with_statement

import os, csv, time, random, math, pickle
import java.io as io
import org.python.util as util

from sys import stdout
from func.nn.backprop import BackPropagationNetworkFactory
from func.nn.activation import LogisticSigmoid
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

def normalize_data(reader_list):
    for i in xrange(len(reader_list[0]) - 1):
        x = [float(row[i]) for row in reader_list]
        mean = sum(x)/len(x)
        variance = sum([xx ** 2 for xx in x])/float(len(x)) - mean ** 2

        for j in xrange(len(reader_list)):
            reader_list[j][i] = math.ceil((x[j] - mean)/(variance ** 0.5) *
                                          1000)/1000

def maybe_serialize(file, force=False):
    serialized_file = os.path.splitext(file)[0] + '.ser'

    if not os.path.isfile(serialized_file) or force:
        stdout.write("Serializing Data-Set...\n\n")

        with open(file, "r") as pima:
            reader_list = list(csv.reader(pima))

            stdout.write("Some sample, un-shuffled data: \n%s\n\n" % reader_list[:3])

            normalize_data(reader_list)
            random.shuffle(reader_list)

            number_of_instances = len(reader_list)
            train_instances = []
            for row in reader_list[:int(number_of_instances *
                                                TRAIN_TEST_SPLIT_RATIO)]:
                instance = Instance([float(value) for value in row[:-1]])
                instance.setLabel(Instance(0 if float(row[-1]) == -1 else 1))
                train_instances.append(instance)

            test_instances = []
            for row in reader_list[int(number_of_instances *
                                               TRAIN_TEST_SPLIT_RATIO):]:
                instance = Instance([float(value) for value in row[:-1]])
                instance.setLabel(Instance(0 if float(row[-1]) == -1 else 1))
                test_instances.append(instance)

            stdout.write("Some sample, shuffled training data (after "
                         "normalization): "
                         "\n%s\n\n" % train_instances[:3])
            stdout.write("Some sample, shuffled test data (after "
                         "normalization): \n%s\n\n" %
                         test_instances[:3])

            stdout.write("Train Data\tTest Data\n")
            stdout.write("%s\t\t%s\n" % (len(train_instances),
                                       len(test_instances)))

            save = {
                TRAIN: train_instances,
                TEST: test_instances,
            }

            outFile = io.FileOutputStream(serialized_file)
            outStream = io.ObjectOutputStream(outFile)

            outStream.writeObject(save)
            outFile.close()
    else:
        stdout.write("Serialized file for data-set found.\n")

    return serialized_file

def train(oa, network, oaName, instances, measure, surpress_output=False, TRAINING_ITERATIONS = 1500):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """

    if not surpress_output:
        print "\nError results for %s every 100 " \
              "iterations\n---------------------------" % (oaName,)

    for i in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        if not surpress_output and i % 100 == 0:
            print "%0.03f" % error

def get_instances(serialized_file, type):
    inFile = io.FileInputStream(serialized_file)
    inStream = util.PythonObjectInputStream(inFile)

    return inStream.readObject()[type]

def evaluate_on_instances(network, instances, type, surpress_output=False):
    correct = 0
    incorrect = 0

    start = time.time()
    for instance in instances:
        network.setInputValues(instance.getData())
        network.run()

        predicted = instance.getLabel().getContinuous()
        actual = network.getOutputValues().get(0)

        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1

    end = time.time()
    query_time = end - start

    if not surpress_output:
        stdout.write("\n%s Results: \nCorrectly classified %d "
                     "instances." % (type, correct))
        stdout.write("\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0))
        stdout.write("\nQuery time: %0.03f seconds\n" % query_time)

    else:
        return float(correct)/(correct+incorrect)*100.0


def main():
    """Run algorithms on the pima dataset."""
    serialized_file = maybe_serialize(INPUT_FILE)
    train_instances = get_instances(serialized_file, TRAIN)

    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]

    for _ in oa_names:
        classification_network = factory.createClassificationNetwork([
            INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER], LogisticSigmoid())
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    learning_curve_data = {}
    learning_curve_file = "learning_curve_data.pickle"

    num_iterations_data = {}
    num_iterations_file = "num_iterations_data.pickle"

    for i in xrange(len(oa_names)):
        stdout.write("\n\nTraining and Testing Neural Network using %s\n\n" %
                     oa_names[i])

        start = time.time()
        train(oa[i], networks[i], oa_names[i], train_instances, measure)
        end = time.time()

        training_time = end - start
        stdout.write("\nTraining completed in %0.03f seconds\n" %
                     training_time)

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        evaluate_on_instances(networks[i], train_instances, "Training")
        test_instances = get_instances(serialized_file, TEST)
        evaluate_on_instances(networks[i], test_instances, "Testing")

        if not os.path.isfile(learning_curve_file):
            stdout.write("\n\nCalculating learning curve data...\n\n")

            x = xrange(38, len(train_instances), 38)
            train_accuracy = []
            test_accuracy = []

            for item in x:
                train(oa[i], networks[i], oa_names[i], train_instances[:item],
                      measure, surpress_output=True)
                optimal_instance = oa[i].getOptimal()
                networks[i].setWeights(optimal_instance.getData())

                train_accuracy.append(evaluate_on_instances(networks[i],
                                                        train_instances[:item],
                                      "Training", surpress_output=True))
                test_accuracy.append(evaluate_on_instances(networks[i],
                                                        test_instances, "Testing",
                                      surpress_output=True))

            learning_curve_data[oa_names[i]] = [train_accuracy, test_accuracy]

        if not os.path.isfile(num_iterations_file):
            stdout.write("\n\nCalculating num iterations data...\n\n")

            x = xrange(100, 1600, 150)
            train_accuracy = []
            test_accuracy = []

            for item in x:
                train(oa[i], networks[i], oa_names[i], train_instances,
                      measure, surpress_output=True, TRAINING_ITERATIONS=item)
                optimal_instance = oa[i].getOptimal()
                networks[i].setWeights(optimal_instance.getData())

                train_accuracy.append(evaluate_on_instances(networks[i],
                                                            train_instances,
                                                            "Training",
                                                            surpress_output=True))
                test_accuracy.append(evaluate_on_instances(networks[i],
                                                           test_instances, "Testing",
                                                           surpress_output=True))

            num_iterations_data[oa_names[i]] = [train_accuracy, test_accuracy]

    if not os.path.isfile(learning_curve_file):
        with open(learning_curve_file, 'wb') as file:
            pickle.dump(learning_curve_data, file, pickle.HIGHEST_PROTOCOL)
            stdout.write("\nLearning Curves Data saved.\n")

    else:
        stdout.write("\nLearning Curves Data found.\n")

    if not os.path.isfile(num_iterations_file):
        with open(num_iterations_file, 'wb') as file:
            pickle.dump(num_iterations_data, file, pickle.HIGHEST_PROTOCOL)
            stdout.write("\nNum Iterations Data saved.\n")
    else:
        stdout.write("\nNum Iterations Data found.\n")



if __name__ == "__main__":
    INPUT_FILE = os.path.join('.', "pima.txt")

    INPUT_LAYER = 8
    HIDDEN_LAYER = 10
    OUTPUT_LAYER = 1
    TRAIN_TEST_SPLIT_RATIO = 0.7
    TRAIN = 'train'
    TEST = 'test'

    main()

