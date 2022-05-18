import collections
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os
import csv

def func_1():
    return


def func_2():
    return

s = 50
m = 2201
start = 26120029
end = 26125038
# start = 0
# end = 50001
variables_quality = np.array([])
variables_positions = np.array([])
variables_id = np.array([])
breeding_rate = 0.8
mutation_rate = 0.2


def initial_population(population_size=200, list_of_functions=[func_1, func_2]):
    global m
    population = np.zeros((population_size, m + len(list_of_functions)))
    for i in range(population_size):
        nums = list(range(population.shape[1]))
        random.shuffle(nums)
        size = int((end - start)/s)
        nums = nums[0:size]
        for num in nums:
            population[i, num] = 1
        # for j in range(m):
        #     population[i, j] = random.randint(0, 1)
        values = np.array(population[i, 0:population.shape[1] - len(list_of_functions)])
        for k in range(1, len(list_of_functions) + 1):
            population[i, -k] = list_of_functions[-k](values)
        print(i)
    print('initial')
    return population


def dominance_function(solution_1, solution_2, number_of_functions=2):
    count = 0
    dominance = True
    for k in range(1, number_of_functions + 1):
        if solution_1[-k] <= solution_2[-k]:
            count = count + 1
    if count == number_of_functions:
        dominance = True
    else:
        dominance = False
    print('dominance')
    return dominance


def raw_fitness_function(population, number_of_functions=2):
    strength = np.zeros((population.shape[0], 1))
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j:
                if dominance_function(solution_1=population[i, :], solution_2=population[j, :],
                                      number_of_functions=number_of_functions):
                    strength[i, 0] = strength[i, 0] + 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j:
                if dominance_function(solution_1=population[i, :], solution_2=population[j, :],
                                      number_of_functions=number_of_functions):
                    raw_fitness[j, 0] = raw_fitness[j, 0] + strength[i, 0]
    print('raw fitness')
    return raw_fitness


def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j]) ** 2 + distance
    return distance ** (1 / 2)


def fitness_calculation(population, raw_fitness, number_of_functions=2):
    k = int(len(population) ** (1 / 2)) - 1
    fitness = np.zeros((population.shape[0], 1))
    distance = np.zeros((population.shape[0], population.shape[0]))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j:
                x = np.copy(population[i, population.shape[1] - number_of_functions:])
                y = np.copy(population[j, population.shape[1] - number_of_functions:])
                distance[i, j] = euclidean_distance(x=x, y=y)
    for i in range(0, fitness.shape[0]):
        distance_ordered = (distance[distance[:, i].argsort()]).T
        fitness[i, 0] = raw_fitness[i, 0] + 1 / (distance_ordered[i, k] + 2)
    print('fitness')
    return fitness


def sort_population_by_fitness(population, fitness):
    idx = np.argsort(fitness[:, -1])
    fitness_new = np.zeros((population.shape[0], 1))
    population_new = np.zeros((population.shape[0], population.shape[1]))
    for i in range(0, population.shape[0]):
        fitness_new[i, 0] = fitness[idx[i], 0]
        for k in range(0, population.shape[1]):
            population_new[i, k] = population[idx[i], k]
    print('sorted')
    return population_new, fitness_new


def roulette_wheel(fitness_new):
    fitness = np.zeros((fitness_new.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i, 0] = 1 / (1 + fitness[i, 0] + abs(fitness[:, 0].min()))
    fit_sum = fitness[:, 0].sum()
    fitness[0, 1] = fitness[0, 0]
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = (fitness[i, 0] + fitness[i - 1, 1])
    for i in range(0, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1] / fit_sum
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if random <= fitness[i, 1]:
            ix = i
            break
    print('wheel')
    return ix


def breeding(population, fitness, list_of_functions=[func_1, func_2]):
    global m
    offspring = np.zeros((population.shape[0], population.shape[1]))
    b_offspring = 0
    for i in range(0, offspring.shape[0] - 1, 2):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]

        probability = random.random()
        # if probability < breeding_rate:
        point = random.randrange(m)
        offspring[i] = np.concatenate([population[parent_1][0:point], population[parent_2][point:]])
        offspring[i + 1] = np.concatenate([population[parent_2][0:point], population[parent_1][point:]])
        # else:
        #     offspring[i] = population[parent_1]
        #     offspring[i + 1] = population[parent_2]

        for k in range(1, len(list_of_functions) + 1):
            offspring[i, -k] = list_of_functions[-k](offspring[i, 0:offspring.shape[1] - len(list_of_functions)])
    print('breeded')
    return offspring


def mutation(offspring, mutation_rate=0.1, list_of_functions=[func_1, func_2]):
    for i in range(0, offspring.shape[0]):
        probability = random.random()
        if probability < mutation_rate:
            pos = random.randrange(offspring.shape[1] - len(list_of_functions))
            offspring[i][pos] = 1 - offspring[i][pos]
            for k in range(1, len(list_of_functions) + 1):
                offspring[i, -k] = list_of_functions[-k](offspring[i, 0:offspring.shape[1] - len(list_of_functions)])
    print('mutated')
    return offspring


def strength_pareto_evolutionary_algorithm_2(population_size=5, archive_size=5, list_of_functions=[func_1, func_2], generations=50):
    count = 0
    population = initial_population(population_size=population_size, list_of_functions=list_of_functions)
    archive = initial_population(population_size=archive_size, list_of_functions=list_of_functions)
    while count <= generations:
        print("Generation = ", count)
        population = np.vstack([population, archive])
        raw_fitness = raw_fitness_function(population, number_of_functions=len(list_of_functions))
        fitness = fitness_calculation(population, raw_fitness, number_of_functions=len(list_of_functions))
        population, fitness = sort_population_by_fitness(population, fitness)
        population, archive, fitness = population[0:population_size, :], population[0:archive_size, :], fitness[
                                                                                                        0:archive_size,
                                                                                                        :]
        population = breeding(population, fitness, list_of_functions=list_of_functions)
        population = mutation(population, list_of_functions=list_of_functions)
        count = count + 1
        print(count)

    return archive


######################## Применение ####################################

# функция отклонения от заданного расстояния
def f1_dist(variables_values: np.array):
    global start, end, variables_positions, s
    values = np.concatenate([[1], variables_values, [1]])
    positions = np.concatenate([[start], variables_positions, [end]])
    d = np.zeros((len(values), len(values)))
    for i in range(len(values)):
        if values[i] == 1 and i < len(values) - 1:
            j = i + 1
            while j < len(values) - 1 and values[j] == 0:
                j += 1
            d[i][j] = abs(s - abs(positions[j] - positions[i]))
            d[j][i] = d[i][j]

    sum_double = 0
    for i in range(len(variables_values) + 1):
        for j in range(i + 1, len(variables_values) + 2):
            sum_double += d[i][j]

    f1 = 1 / (1 + sum(variables_values)) * sum_double
    print("f1")
    return f1


# функция качества
def f2_guality(variables_values: np.array):
    global variables_quality
    a = (sum(variables_values))
    if a == 0: a = 1
    f2 = - 1 / a * sum([variables_quality[i] * variables_values[i] for i in range(len(variables_values))])
    print("f2")
    return f2


def read_data(file: str):
    global variables_positions, variables_quality, variables_id, start, end
    with open(file, newline='') as File:
        reader = csv.reader(File, delimiter='\t')
        for row in reader:
            if row[2] in variables_positions or int(row[2]) < start + 1 or int(row[2]) > end - 1:
                continue
            if row[0] == "":
                continue
            variables_id = np.append(variables_id, int(row[0]))
            variables_positions = np.append(variables_positions, int(row[2]))
            variables_quality = np.append(variables_quality, float(row[1]))
    z = zip(variables_positions, variables_quality, variables_id)
    sor = sorted(z, key=lambda tup: tup[0])
    variables_positions = [int(x[0]) for x in sor]
    variables_quality = [float(x[1]) for x in sor]
    variables_id = [x[2] for x in sor]
    print("read_data")


def find_result():
    global variables_quality
    spea_2_snp = strength_pareto_evolutionary_algorithm_2(population_size=100, archive_size=100,
                                                          list_of_functions=[f1_dist, f2_guality], generations=10)
    with open("real.txt", mode="w", encoding='utf-8') as output_file:
        for r in spea_2_snp:
            string = ''
            for i in range(len(r)):
                if r[i] == 1:
                    string += str(variables_id[i]) + " "
                else:
                    string += "0 "
            string += str(r[-2])
            string += str(r[-1])
            output_file.write(string + "\n")

    # x = np.zeros(len(variables_quality))
    # i = 0
    # for i in range(len(variables_quality)):
    #     if variables_quality[i] == 1:
    #         x[i] = 1

    func_1_values = spea_2_snp[:, -2]
    func_2_values = spea_2_snp[:, -1]
    ax1 = plt.figure(figsize=(15, 15)).add_subplot(111)
    plt.xlabel('Function 1', fontsize=12)
    plt.ylabel('Function 2', fontsize=12)
    ax1.scatter(func_1_values, func_2_values, c='red', s=25, marker='o', label='SPEA-2')
    # ax1.scatter(f1_dist(x), f2_guality(x), c='blue', s=25, marker='o', label='reference')
    plt.legend(loc='upper right')
    plt.show()


def main():
    read_data('results.csv')
    find_result()


if __name__ == '__main__':
    main()