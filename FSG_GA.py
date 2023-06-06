#Import part
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
#Auxiliary GA operations
def _utils_constraints(g, min, max):
    if max and g > max:
        g = max
    if min and g < min:
        g = min
    return g
def crossover_blend(g1, g2, alpha, min = None, max = None):
    shift = (1. + 2. * alpha) * random.random() - alpha
    new_g1 = (1. - shift) * g1 + shift * g2
    new_g2 = shift * g1 + (1. - shift) * g2
    return _utils_constraints(new_g1, min, max),_utils_constraints(new_g2, min, max)
def mutate_gaussian(g, mu, sigma, min = None, max = None):
    mutated_gene = g + random.gauss(mu, sigma)
    return _utils_constraints(mutated_gene, min, max)
def select_tournament(population, tournament_size):
    new_oﬀspring = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in range(tournament_size)]
        new_oﬀspring.append(max(candidates, key = lambda ind:ind.ﬁtness))
    return new_oﬀspring
def func(x):
    a,b,c,bv=100, 473, 0.35,1.5
    kbz = 8.617385e-5
    #return a*np.exp(1.0+c/kbz/x*((x-b)/b)-((x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)-2.0*kbz*b/c)
    #return 4*a*np.exp(c/kbz/x*((x-b)/b))*(((x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+2.0*kbz*b/c)**(-2)
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))


def get_best(population):
    best = population[0]
    for ind in population:
        if ind.ﬁtness > best.ﬁtness:
            best = ind
    return best
def plot_population(population, number_of_population):
    best = get_best(population)
    x = np.linspace(0, 900)
    plt.plot(x, func(x), '--' , color = 'blue')
    plt.plot([ind.get_gene() for ind in population], [ind.ﬁtness for ind in
    population], 'o' , color = 'orange')
    plt.plot([best.get_gene()], [best.ﬁtness], 's' , color = 'green')
    plt.title(f"Generation number {number_of_population}")
plt.show()
plt.close()
#Individual class
class Individual:
    def __init__(self, gene_list: List[ﬂoat]) -> None:
        self.gene_list = gene_list
        self.ﬁtness = func(self.gene_list[0])
    def get_gene(self):
        return self.gene_list[0]
    @classmethod
    def crossover(cls, parent1, parent2):
        child1_gene, child2_gene = crossover_blend(parent1.get_gene(),
        parent2.get_gene(), 1, -10, 10)
        return Individual([child1_gene]), Individual([child2_gene])
    @classmethod
    def mutate(cls, ind):
        mutated_gene = mutate_gaussian(ind.get_gene(), 0, 1, -10, 10)
        return Individual([mutated_gene])
    @classmethod
    def select(cls, population):
        return select_tournament(population, tournament_size = 3)
    @classmethod
    def create_random(cls):
        return Individual([random.randrange(0, 900)])
#GA flow
random.seed(16)
# random.seed(16) # local maximum
POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = .8
MUTATION_PROBABILITY = .1
MAX_GENERATIONS = 10
ﬁrst_population = [Individual.create_random() for _ in
range(POPULATION_SIZE)]
plot_population(ﬁrst_population, 0)
generation_number = 0
population = ﬁrst_population.copy()
while generation_number < MAX_GENERATIONS:
    generation_number += 1
# SELECTION
oﬀspring = Individual.select(population)
# CROSSOVER
crossed_oﬀspring = []
for ind1, ind2 in zip(oﬀspring[::2], oﬀspring[1::2]):
    if random.random() < CROSSOVER_PROBABILITY:
        kid1, kid2 = Individual.crossover(ind1, ind2)
        crossed_oﬀspring.append(kid1)
        crossed_oﬀspring.append(kid2)
    else:
        crossed_oﬀspring.append(ind1)
        crossed_oﬀspring.append(ind2)
# MUTATION
mutated_oﬀspring = []
for mutant in crossed_oﬀspring:
    if random.random() < MUTATION_PROBABILITY:
        new_mutant = Individual.mutate(mutant)
        mutated_oﬀspring.append(new_mutant)
    else:
        mutated_oﬀspring.append(mutant)
        population = mutated_oﬀspring.copy()
        plot_population(population, generation_number)
     