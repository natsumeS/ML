import numpy as np
import random

from myML.ga.gene import Gene


class Generation:
    def __init__(self, num_gene: int, gene_type: type):
        self.num_gene = num_gene
        self.genes = []
        self.gene_type = gene_type
        for i in range(self.num_gene):
            self.genes.append(gene_type())

    def selection_roulette(self):
        # calculate the probability for selecting each genes.
        fit_list = []
        for gene in self.genes:
            fitness = gene.get_fitness()
            fit_list.append(fitness)

        # tmp variable to turn fitness into positive number
        min_v = np.min(fit_list)
        fit_list = fit_list + min_v
        fitness_sum = np.sum(fit_list)
        fit_list = fit_list / fitness_sum

        # update generations
        self.genes = list(np.random.choice(self.genes, p=fit_list, size=self.num_gene, replace=True))

    def selection_tournament(self, size: int = 2):
        prod = self.num_gene // size
        extra = self.num_gene % size
        new_generations = []
        # shuffle
        random.shuffle(self.genes)

        # start_loop
        group_counter = 0
        for i in range(prod):
            # sampling
            samp = []
            tmp_size = size
            for j in range(size):
                samp.append(self.genes.pop())
            if group_counter < extra:
                samp.append(self.genes.pop())
                tmp_size += 1

            # sort gene
            samp.sort(key=lambda g: g.get_fitness(), reverse=True)

            # copy best gene
            for j in range(tmp_size):
                new_generations.append(self.gene_type(samp[0].content))

            # increment
            group_counter += 1

        # set new generation
        self.genes = new_generations

    def crossover(self, func=None):
        if func is None:
            raise Exception("func:callback should be set")
        # shuffle
        random.shuffle(self.genes)
        # crossover
        generations = []
        for i in range(int(self.num_gene / 2)):
            gene1, gene2 = func(self.genes.pop(), self.genes.pop())
            generations.append(gene1)
            generations.append(gene2)

        # in case that num_gene is odd
        if self.num_gene % 2 == 1:
            generations.append(self.genes.pop())
        # update genes
        self.genes = generations

    def mutation(self, **kwords):
        for gene in self.genes:
            gene.mutation(**kwords)

    def evaluate(self) -> float:
        score = 0.0
        for gene in self.genes:
            score += gene.get_fitness()
        return score / self.num_gene

    def get_best(self) -> (float, Gene):
        # sort
        self.genes.sort(key=lambda g: g.get_fitness(), reverse=True)
        return self.genes[0].get_fitness(), self.genes[0].content
