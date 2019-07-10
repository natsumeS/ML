import numpy as np


class Gene:
    def __init__(self, content=None, **initialize_args):
        self.content = content
        if self.content is None:
            self.initialize(**initialize_args)
        self.fitness = None

    def initialize(self, **args):
        raise Exception("initialize")

    def evaluate(self) -> float:
        raise Exception("fitness error")

    def get_fitness(self) -> float:
        if self.fitness is None:
            self.fitness = self.evaluate()
        return self.fitness

    def mutation(self):
        raise Exception("override mutation method!!")

    def set_content(self, content):
        self.content = content
        self.fitness = None


class RGene(Gene):
    low_arr = None
    high_arr = None

    def __init__(self, content=None):
        super().__init__(content)

    def initialize(self):
        self.content = np.random.uniform(low=self.__class__.low_arr, high=self.__class__.high_arr)

    def mutation(self, p=0.01, delta=0.2):
        content = []
        for i, ele in enumerate(self.content):
            if np.random.rand() > p:
                content.append(ele)
                continue
            content.append(np.random.uniform(low=max(self.low_arr[i], ele - delta),
                                             high=min(self.high_arr[i], ele + delta)))
        self.set_content(np.array(content))
