#!/usr/bin/env python
import argparse
import functools
import math
import random

import numpy
import matplotlib.pyplot as plt
import scipy.special

from config import config
import PID

MAX_K = 10
MAX_MUTATION = 3


class GAConfigError(Exception):
    '''
    Error representing incorrect configuration of the genetic algorithm.
    '''


class Genome(object):
    '''
    Class representing a single genome.
    '''
    def __init__(self, num_genes):
        self._num_genes = num_genes
        self._genes = [random.random() * MAX_K for _ in range(num_genes)]
        self._fitness = 0.0

    @classmethod
    def from_genes(cls, genes):
        inst = cls.__new__(cls)
        inst._num_genes = len(genes)
        inst._genes = genes
        inst._fitness = 0.0
        return inst

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes):
        self._genes = genes

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

    @property
    def kp(self):
        return self.genes[0]

    @property
    def ki(self):
        return self.genes[1]

    @property
    def kd(self):
        return self.genes[2]

    def crossover(self, genome):
        '''
        Takes another genome, selects midpoint and swaps the ends of each
        genome.

        @param genome - another genome to crossover with
        @returns a tuple of two genome babies
        '''
        value = random.random()
        if value < 1.0/5:
            return Genome.from_genes([self.kp, self.ki, genome.kd])
        elif value < 2.0/5:
            return Genome.from_genes([self.kp, genome.ki, self.kd])
        elif value < 3.0/5:
            return Genome.from_genes([self.kp, genome.ki, genome.kd])
        elif value < 4.0/5:
            return Genome.from_genes([genome.kp, self.ki, self.kd])
        elif value < 4.0/5:
            return Genome.from_genes([genome.kp, self.ki, genome.kd])
        else:
            return Genome.from_genes([genome.kp, genome.ki, self.kd])

    def mutate(self, mutation_rate):
        '''
        Mutate this genome.

        @param mutation_rate - probability of this genome to be mutated
        '''
        for i, gene in enumerate(self.genes):
            if random.random() <= mutation_rate:
                # Generate a value between -MAX_MUTATION and MAX_MUTATION
                change = MAX_MUTATION * 2 * random.random() - MAX_MUTATION
                new_value = self.genes[i] + change
                self.genes[i] = new_value if new_value >= 0 else random.random() * MAX_K

    def __str__(self):
        return "KP: {0}, KI: {1}, KD: {2}".format(
               self.kp, self.ki, self.kd)


class GAPID(object):
    '''
    Genetic algorithm made for optimizing PID loop parameters.
    '''
    def __init__(self, cross_rate, mutation_rate, population_size, fitness_func):
        '''
        Initializes instance of GAPID.

        @param cross_rate - probability of crossing genes
        @param mutation_rate - probability of gene mutation
        @param population_size - size of the population to breed
        @param fitness_func - function to test an individual fitness
                               signature: fitness_func(Genome_instance)
        '''
        self._cross_rate = cross_rate
        self._mutation_rate = mutation_rate
        self._population_size = population_size
        self._fitness_func = fitness_func
        self._genomes = [Genome(num_genes=3) for _ in range(population_size)]
        self._generation = 0
        self._fittest_genome = None
        self._total_fitness = 0.0

    @property
    def fittest_genome(self):
        '''
        Fittest genome from the last generation.
        '''
        return self._fittest_genome

    def _create_offspring(self):
        '''
        This method picks two parents using whatever selection algorithm was
        configured, and creates their baby.

        @returns a single genome
        '''
        mom = self._select_parent(10)
        dad = self._select_parent(10)

        def do_crossover():
            return random.random() <= self._cross_rate

        if do_crossover() and mom != dad:
            baby = dad.crossover(mom)
        else:
            baby = random.choice((dad, mom))

        baby.mutate(self._mutation_rate)
        return baby

    def _roulette_wheel_select(self):
        '''
        Select a gene from the population with a higher probability of picking
        a more fit individual.

        @returns a single genome
        '''
        wheel_slice = random.random() * self._total_fitness
        total = 0.0

        for gene in self._genomes:
            total += gene.fitness
            if total > wheel_slice:
                return gene

    def _elite_gene_selection(self, N):
        '''
        This was supposed to be an elitist selection algorithm, but most of the
        time it fails to converge on a good solution due to a high likeleyhood
        of getting stuck in the local minima due always picking the best of the
        crop from the last generation. Need to rethink this method to retain
        some elite genes, but still pick from the larger population to increase
        diversity.

        @param N - number of individual to sample from the population
        @returns a single genome
        '''
        N = min(N, self._population_size)
        def numeric_compare(g1, g2):
            return int(g2.fitness - g1.fitness)

        sorted_genomes = sorted(self._genomes, cmp=numeric_compare)
        chosen_one = random.randrange(N)
        return sorted_genomes[chosen_one]

    def _tournament_selection(self, N):
        '''
        Select the parent based on the tournament selection algorithm.

        @param N - number of individuals to sample
        @returns a single genome
        '''
        best_so_far = 0.0
        chosen_one = -1
        for i in range(N):
            this_try = random.randrange(0, self._population_size-1)
            if self._genomes[this_try].fitness > best_so_far:
                chosen_one = this_try
                best_so_far = self._genomes[this_try].fitness

        return self._genomes[chosen_one]

    def _select_parent(self, N):
        '''
        Use a selection algorithm to pick a parent.
        @param N - number of individuals to sample
        @returns a single genome
        '''
        selection_alg = config.get('selection_alg', 'tournament')
        if selection_alg == 'tournament':
            return self._tournament_selection(N)
        elif selection_alg == 'elite':
            return self._elite_gene_selection(N)
        elif selection_alg == 'roulette':
            return self._roulette_wheel_select()
        else:
            raise GAConfigError("{0} selection algorithm is not supported."
                                .format(selection_alg))

    def epoch(self):
        '''
        The workhorse of the GA. This method updates fitness scores of the
        population then creates a new population of genomes using the
        Selection, Crossover and Mutation operators.
        '''
        self.update_fitness_scores()

        new_generation = []

        while len(new_generation) < self._population_size:
            new_generation.append(self._create_offspring())

        self._genomes = new_generation
        self._generation += 1

    def update_fitness_scores(self):
        '''
        Test each individual in the population to calculate their fitness
        score. This method sets the fittest_genome property.
        '''
        self._total_fitness = 0.0
        best_score = 0
        self._fittest_genome = None

        for genome in self._genomes:
            genome.fitness = self._fitness_func(genome)
            self._total_fitness += genome.fitness

            if genome.fitness > best_score:
                best_score = genome.fitness
                self._fittest_genome = genome


def ideal_trajectory_curve():
    '''
    Calculate the ideal trajectory of the PID loop - currently using the
    incomplete gamma function.

    @returns numpy array
    '''
    scale = config['search_target']
    num_samples = config['num_samples']
    xs = numpy.linspace(0, 30, num_samples)
    ys = scipy.special.gammainc(1, xs) * scale
    return ys


def physics_update(current_position, velocity, gravity, dt):
    '''
    @param current_position - current system value
    @param velocity - current velocity
    @param gravity - force of gravity
    @param dt - time delta
    @return (new_velocity, new_position)
    '''
    new_position = current_position + (velocity * dt)
    new_velocity = velocity + (gravity * dt)
    return new_velocity, new_position


def simulate(genome, dt):
    '''
    Simulate the flight of the flappy bird.

    @param genome - genome representing PID parameters
    @param dt - time delta between update calls
    @return distance from the target at each update call
    '''
    min_velocity = config['min_velocity']
    max_velocity = config['max_velocity']
    gravity = config['gravity']
    pid = PID.PID(*genome.genes, min_cor=min_velocity, max_cor=max_velocity)
    pid.target = config['search_target']

    position = 0.0
    velocity = 0.0
    distances = []

    num_samples = config['num_samples']
    for step in range(num_samples):
        velocity, position = physics_update(position, velocity, gravity, dt)
        velocity = pid.make_correction(position, dt)
        distances.append(position)

    return numpy.array(distances)


def pid_fitness_func(genome, dt):
    '''
    This function is used by the Genetic Algorithm to optimize the PID loop to
    find optimal parameters.

    @param genome - genome representing PID parameters
    @param dt - time delta
    '''
    distances = simulate(genome, dt)
    trajectory = ideal_trajectory_curve()

    distance_sum = numpy.sum(numpy.abs(distances - trajectory))
    return 1 / numpy.sqrt(distance_sum)


def draw_plot(fittest_genome, ideal_trajectory, genome_trajectory):
    '''
    Plots the ideal trajectory, best generation and error.

    @param ideal_trajectory - trajectory being optimized to
    @param genome_trajectory - trajectory of a candidate genome
    '''
    error = numpy.abs(ideal_trajectory - genome_trajectory)

    plt.suptitle(str(fittest_genome), fontsize=12)
    ax = plt.subplot(111)
    ax.plot(ideal_trajectory, label="Target")
    ax.plot(genome_trajectory, label="Gene")
    ax.plot(error, label="Error")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(.5, -0.05), ncol=3)


def main():
    parser = argparse.ArgumentParser(description="Flappy PID loop controller optimizer")
    parser.add_argument("-o",
                        "--optimize",
                        help="Perform optimization search.",
                        action="store_true",
                        default=False)
    parser.add_argument("-g",
                        "--generations",
                        help="Number of generations to run search (100)",
                        default=100)
    parser.add_argument("-n",
                        "--num_genes",
                        help="Number of genes in each generation (100)",
                        default=100)
    parser.add_argument("-p",
                        "--KP",
                        help="Proportional gain (1.0)",
                        type=float,
                        default=1.0)
    parser.add_argument("-i",
                        "--KI",
                        help="Integral gain (1.0)",
                        type=float,
                        default=1.0)
    parser.add_argument("-d",
                        "--KD",
                        help="Derivative gain (1.0)",
                        type=float,
                        default=1.0)

    args = parser.parse_args()

    dt = 0.01666666666
    ideal_trajectory = ideal_trajectory_curve()
    if args.optimize:

        fitness_func = functools.partial(pid_fitness_func, dt=dt)
        gapid = GAPID(cross_rate=0.85,
                      mutation_rate=0.02,
                      population_size=args.num_genes,
                      fitness_func=fitness_func)

        for g in range(args.generations):
            gapid.epoch()
            fittest = gapid.fittest_genome
            print "Generation: {0} Best_score: {1} total_fitness: {2} best_genome: {3}".format(
                  gapid._generation, fittest.fitness, gapid._total_fitness, fittest)
            best_gene_trajectory = simulate(gapid._fittest_genome, dt)

            draw_plot(gapid._fittest_genome, ideal_trajectory, best_gene_trajectory)

            plt.savefig("/run/shm/generation_%s.png" % g)
            plt.cla()
            plt.clf()
    else:
        genome = Genome.from_genes([args.KP, args.KI, args.KD])
        gene_trajectory = simulate(genome, dt)

        draw_plot(genome, ideal_trajectory, gene_trajectory)
        plt.show()


if __name__ == "__main__":
    main()
