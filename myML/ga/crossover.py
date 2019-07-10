import numpy as np
from myML.ga.gene import RGene


def arithmetical_crossover(gene1: RGene, gene2: RGene, *, lam: float = 0.2) -> (RGene, RGene):
    arr1 = lam * gene1.content + (1 - lam) * gene2.content
    arr2 = (1 - lam) * gene1.content + lam * gene2.content
    return gene1.__class__(arr1), gene1.__class__(arr2)


def average_crossover(gene1: RGene, gene2: RGene) -> (RGene, RGene):
    # generate three gene
    arr1 = (3 * gene1.content - gene2.content) / 2.0
    arr2 = (3 * gene2.content - gene2.content) / 2.0
    arr3 = (gene1.content + gene2.content) / 2.0
    genes = [gene1.__class__(arr1), gene1.__class__(arr2), gene1.__class__(arr3)]

    # pick up best two genes
    genes.sort(key=lambda g: g.get_fitness(), reverse=True)
    return genes[0], genes[1]


def blend_crossover(gene1: RGene, gene2: RGene, *, alpha: float = 0.2) -> (RGene, RGene):
    cls = gene1.__class__

    # calculate alpha range
    alpha_range = alpha * np.abs(gene1.content - gene2.content)
    arr1, arr2 = [], []
    for i, ele1, ele2 in enumerate(gene1.content, gene2.content):
        # calcurate boundary of sample range
        min_ele = max(cls.low_arr[i], min(ele1, ele2) - alpha_range[i])
        max_ele = min(cls.high_arr[i], max(ele1, ele2) + alpha_range[i])

        # sampling data
        samp = np.random.uniform(low=min_ele, high=max_ele, size=2)
        arr1.append(samp[0])
        arr2.append(samp[1])

    return cls(np.array(arr1)), cls(np.array(arr2))


if __name__ == '__main__':
    class myGene(RGene):
        low_arr = [-1, -1, -1]
        high_arr = [1, 1, 1]

        def __init__(self, content=None):
            super().__init__(content)

        def evaluate(self) -> float:
            return np.sum(np.square(self.content))


    gene1 = myGene()
    gene2 = myGene()
    print(gene1.content, gene2.content)
    average_crossover(gene1, gene2)
    print(np.abs(np.array([1, -2, 3])))
    # gene3 = gene2.copy()
    # print(gene3)
    # print(gene3.content)
