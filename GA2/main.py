from deap import base, algorithms
from deap import creator
from deap import tools
from prettytable import PrettyTable
import random
import numpy as np

# константы задачи
genes_lenght = 8    # количество генов в особи

# константы генетического алгоритма
population_size=10; # количество особей в популяции
p_cross = 0.9       # вероятность скрещивания
p_mutation = 0.3
MAX_GENERATIONS = 1    # максимальное количество поколений

RANDOM_SEED = 50
random.seed(RANDOM_SEED)
def pretty_output(data,th):
    #определяем данные
    fitness=[]
    td=[]
    for x in data:
        fitness = oneMaxFitness(x)
        td.append(''.join(str(i) for i in x))
        td.append(str(fitness[0]))
    columns=len(th) # Подсчитываем кол-во столбцов
    table=PrettyTable(th) # Определяем таблицу.
    td_data=td[:]
    while td_data:
        table.add_row(td_data[:columns])
        td_data=td_data[columns:]
    print(table)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def oneMaxFitness(individual):
    return sum(individual)/len(individual), # кортеж

toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, genes_lenght)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=population_size)
pretty_output(population, ["Первоначальная популяция", "Приспособленность"])

toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select",  tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/genes_lenght)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=p_cross,
                                        mutpb=p_mutation,
                                        ngen=MAX_GENERATIONS,
                                        stats=stats,
                                        verbose=True)
pretty_output(population, ["Популяция после мутации", "Приспособленность"])
#maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
fitnessValues = [ind.fitness.values[0] for ind in population]  #обновляем значения приспособленности новой популяции
maxFitnessValues = []
maxFitness = max(fitnessValues)
maxFitnessValues.append(maxFitness)
best_index = fitnessValues.index(max(fitnessValues))
print("Лучшая особь = ", *population[best_index], "\n")

