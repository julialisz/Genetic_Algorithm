import random
import csv
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.tools import FigureFactory as FF
import pandas as pd
import scipy

n = random.randint(1000, 2000)
w = random.randint(10000, 20000)
s = random.randint(10000, 20000)
#print([n, w, s])
output_file = 'generated.csv'
input_file = 'generated.csv'

"""
Task generator for the knapsack problem (0.5 points)

The task generator should be implemented as a function:
   generate(n, w, s, output_file)
This function takes the following parameters:
● n - number of objects to choose (int)
● w - maximum carrying capacity of the knapsack (int)
● s - maximum knapsack size (int)
● output_file - name of the file into which the task is to be saved

The generator for a given number of items is to randomly generate weights and sizes of items.
The weight w_i, size s_i and price c_i of the i-th item must meet the following criteria:
● 0 < w_i < 10*w/n
● 0 < s_i < 10*s/n
● 0 < c_i < n
In addition, a set of items must meet the following criteria:
    - sum(from 1 to n) w_i > 2w
    - sum(from 1 to n) s_i > 2s
The generator thus prepared should generate a solution for a random n, w, s from the following ranges:
● 1000 < n < 2000
● 10000 < w < 20000
● 10000 < s < 20000
The file with the solution should be saved in CSV format. The first line of the file
contains numbers n, w, s (separated by a comma). The next lines represent objects. Line
i contains the following numbers (separated by a comma): w_i, s_i, c_i. The generated file
should be attached to the solution package and sent to the e-portal.
"""
def generate(n, w, s, output_file):
    with open(output_file,'w') as f:
        file = csv.writer(f)
        file.writerow([n, w, s])
        sum_wi = 0
        sum_si = 0
        for i in range(n):
            w_i = random.uniform(0, 10*w/n)
            s_i = random.uniform(0, 10*s/n)
            c_i = random.uniform(0, n)
            file.writerow([w_i, s_i, c_i])
            sum_wi += w_i
            sum_si += s_i
        if not(sum_wi > (2*w) and sum_si > (2*s)):
            generate(n, w, s, output_file)


"""
Task loading (0.5 points)

The method of loading a task as a function should be implemented:
    read(input_file)
This function takes a file name in CSV format on the input. The first line of the file contains numbers n, w, s 
(separated by a comma). The next lines represent objects. Line i contains the following numbers (separated by a comma): 
w_i, s_i, c_i.
Suggestion: At the output the method can return a Task class object, in which a structure will be defined to store 
the elements of the knapsack problem.
"""
def read(input_file):
    tasks = []  #[[0.0 for x in range(3)] for y in range(n+1)]
    parameters = [0, 0, 0]
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter=',')
        '''row1 = next(reader)  # gets the first line
        N = int(row1[0])
        W = int(row1[1])
        S = int(row1[2])
        print(N, W, S)
        print(row1)'''
        for row in reader:
            if(row):
                parameters = [round(float(row[0])), round(float(row[1])), round(float(row[2]))]
                tasks.append(parameters)
    #print(tasks)
    #print(tasks[0])
    #print(tasks[0][0])
    return tasks


"""
Creation of a random initial population (0.5 points)

A method should be implemented to create a random initial population for the knapsack problem as a function:
    init_population(n_items, size)
This function takes the following parameters:
● n_items - number of objects
● size - population size
Suggestion: The function can return a Population class object as an output. This class may contain a data structure for 
storing Individual class elements.
"""
def init_population(n_items, size):
    population = []
    for x in range (0, size):
        x_i = []
        for i in range (0, n_items):    #n items in population (length)
            x_i.append(np.random.choice(np.arange(0, 2), p=[0.81, 0.19]))  #p=[0.81, 0.19]
        population.append(x_i)
    #print(population)
    return population

"""
Fitness function (0.5 points)

A method of assessing the fitness of an individual as a function should be implemented:
    evaluate(item, task)
Suggestion: The function takes the individual as defined in the first part of the task (e.g. Task object). This 
function can also be defined as an Individual class function.
The output of the function is the fitness value for the individual. The value of the matching function for a given 
solution is the sum of the values of all objects in the knapsack if the constraints are met, or 0 otherwise.
"""
def evaluate(x, task):
    sum_weight = 0
    sum_size = 0
    fitness = 0
    #print("This is size: " + str(size))
    for i in range(0, len(x)):
        w_i = task[i+1][0]
        s_i = task[i+1][1]
        c_i = task[i+1][2]
        x_i = x[i]
        sum_weight += w_i * x_i
        #print("Weight: " + str(sum_weight))
        sum_size += s_i * x_i
        fitness += c_i * x_i
    if ((sum_weight <= w) and (sum_size <= s)):
        #print("Fitness function: " + str(fitness))
        return fitness
    else:
        return 0


"""
Tournament selection method (0.5 points)

The tournament selection method should be implemented as a function: 
    tournament (population, tournament_size)
The function takes the following parameters:
● population - a Population class object
● tournament_size - a size of the tournament
A simple tournament selection can be made as follows:
● Select k (tournament_size) of individuals at random from the population
● Return the best individual in the tournament
The best individual is the individual with the highest value of the evaluate function.
"""
def tournament (population, tournament_size):
    best = 0
    task = read(input_file)
    for k in range(tournament_size):
        rand = random.randint(0, len(population)-1)
        ind = population[rand]
        if ((best == 0) or (evaluate(ind, task) > evaluate(best, task))):
            best = ind
    #print(best)
    return best


"""
Crossover operator (0.5 points)

The crossover method should be implemented as a function:
    crossover(parent1, parent2, crossover_rate)
A simple crossover method involves selecting a cutting point for the parents' chromosomes. The initial fragments of the 
chromosomes are then swapped in places. This produces a child of a pair of parent1 and parent2.
The crossover_rate parameter is the probability of a crossover occurring. Before the crossover itself, it is necessary 
to simulate whether the crossover is to take place. For this purpose, the real number should be drawn from the range 
[0, 1]. If rand < crossover_rate, then crossover occurs and the child is returned. Otherwise, parent1 is returned.
"""
def crossover(parent1, parent2, crossover_rate):
    cross_point = 500
    parent1 = [str(integer) for integer in parent1]
    parent2 = [str(integer) for integer in parent2]
    rand = random.uniform(0, 1)
    if rand < crossover_rate:
        #print("Cross > rate")
        for i in range(cross_point, len(parent1)):
            parent1[i], parent2[i] = parent2[i], parent1[i]
        parent1 = ''.join(parent2)
        parent1 = [int(char) for char in parent1]
        child = parent1
        return child
    else:
        parent1 = [int(char) for char in parent1]
        return parent1


"""
Mutation operator (0.5 points)

The mutation method should be implemented as a function:
    mutate(individual, mutation_rate)
A simple mutation method computes how many genes will be mutated on the basis of the size of an individual n (number of 
genes - in the case of the problem the number of all available items). In case of n=1000 and mutation_rate=0.01 the 
number of genes that are subject to mutation is: 1000*0.01=10. Therefore 10 positions in the individual
chromosome should be randomly selected and for each position the gene value should be changed to 0 if the current 
value is 1, or 1 if the current value is 0.
"""
def mutate(individual, mutation_rate):
    no_of_positions = int(n * mutation_rate)
    for i in range(0, no_of_positions):
        rand_position = random.randint(0, n-1)
        if(individual[rand_position] == 0):
            individual[rand_position] = 1
        else:
            individual[rand_position] = 0
    #print(individual)


"""
Genetic algorithm (7 points)

The genetic algorithm should be implemented using previously created elements. The
pseudocode to the whole looks like this:
task = read(input_file)
pop = init_population(task.n_items, POP_SIZE)
i = 0
while i < ITERATIONS:
    j = 0
    new_pop = Population() --robimy nową pustą populację
    while j < POP_SIZE:
    parent1 = tournament(pop)
    parent2 = tournament(pop)
    child = crossover(parent1, parent2, CROSSOVER_RATE)
    mutate(child, MUTATION_RATE)
    new_pop.add(child)
    j += 1
pop = new_pop
i += 1
return pop.best()
The algorithm returns the best individual in the last population as a result.
"""
def genetic_algorithm(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size):
    #POP_SIZE = 200
    #ITERATIONS = 30
    #CROSSOVER_RATE = 0.4
    #MUTATION_RATE = 0.01
    #tournament_size = 30
    generate(n, w, s, output_file)
    print("n: " + str(n))
    task = read(input_file)
    pop = init_population(n, POP_SIZE)
    i = 0
    fitness = []
    generations = []
    while (i < ITERATIONS):
        print("Iteration: " + str(i+1))
        j = 0
        new_pop = []
        while(j < POP_SIZE):
            parent1 = tournament(pop, tournament_size)
            parent2 = tournament(pop, tournament_size)
            child = crossover(parent1, parent2, CROSSOVER_RATE)
            mutate(child, MUTATION_RATE)
            new_pop.append(child)
            #fit = evaluate(child, task)
            j += 1
        pop = new_pop
        best_ind = tournament(pop, POP_SIZE)
        #print("Len of individual: " + str(len(best_ind)))
        best_ind_fit = evaluate(best_ind, task)
        fitness.append(best_ind_fit)
        generations.append(i)
        #print(best_ind_fit)
        i += 1
    #print("Length pop: " + str(len(pop)))
    best_ind = tournament(pop, POP_SIZE)
    #print(best_ind)
    #print(len(best_ind))
    best_ind_fit = evaluate(best_ind, task)
    print(best_ind_fit)
    print(fitness)
    print(generations)
    return best_ind


"""generate(n, w, s, output_file)
read(input_file)
pop_size = 100
init_population(n, pop_size)
task = read(input_file)
population = init_population(n, pop_size)

big = 0
zero = 0
for i in range (0, pop_size):
    eval = evaluate(population[i], task)
    if (eval > 0):
        big += 1
    else:
        zero += 1
print("Zero: " + str(zero))
print("Bigger: " + str(big))
best_test = tournament(population, pop_size)
print("TEST tournament evaluation: "+ str(evaluate(best_test, task)))
cross = crossover(population[3], population[49], 0.4)
print("Crossover: " + str(cross))
mutate(cross, 0.2)
print(evaluate(cross, task))"""

#genetic_algorithm(100, 5, 0.4, 0.005, 30)



"""
Analysis of the impact of the crossover probability (1 point)

The impact of the crossover probability on the results should be investigated. For this purpose, a minimum of 3 
different crossover probability values must be selected and the test for each value must be carried out at least 5 
times. The averaged values of the best individuals in subsequent generations should be shown on a chart. The conclusions
of the study should be described in the report.
"""
def crossover_impact():
    def gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size):
        generate(n, w, s, output_file)
        #print("n: " + str(n))
        task = read(input_file)
        pop = init_population(n, POP_SIZE)
        i = 0
        fitness = []
        #generations = []
        while (i < ITERATIONS):
            print("Iteration: " + str(i+1))
            j = 0
            new_pop = []
            while(j < POP_SIZE):
                parent1 = tournament(pop, tournament_size)
                parent2 = tournament(pop, tournament_size)
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                mutate(child, MUTATION_RATE)
                new_pop.append(child)
                j += 1
            pop = new_pop
            best_ind = tournament(pop, POP_SIZE)
            best_ind_fit = evaluate(best_ind, task)
            print("Fitness: " + str(best_ind_fit))
            fitness.append(best_ind_fit)
            #generations.append(i+1)
            i += 1
        return fitness

    POP_SIZE = 100
    ITERATIONS = 20
    CROSSOVER_RATE = 0
    MUTATION_RATE = 0.005
    tournament_size = 30
    #gen = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)[0]
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_0_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    '''df = pd.DataFrame({'x': range(ITERATIONS), 'y': fit_0_average})
    plt.plot('x', 'y', data=df, linestyle='-', marker='o')
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()'''


    CROSSOVER_RATE = 0.3
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_03_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    '''df = pd.DataFrame({'x': range(ITERATIONS), 'y': fit_03_average})
    plt.plot('x', 'y', data=df, linestyle='-', marker='o')
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()'''


    CROSSOVER_RATE = 0.6
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_06_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    '''df = pd.DataFrame({'x': range(ITERATIONS), 'y': fit_06_average})
    plt.plot('x', 'y', data=df, linestyle='-', marker='o')
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()'''


    CROSSOVER_RATE = 0.85
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_085_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    '''df = pd.DataFrame({'x': range(ITERATIONS), 'y': fit_085_average})
    plt.plot('x', 'y', data=df, linestyle='-', marker='o')
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()'''

    df=pd.DataFrame({'x': range(ITERATIONS), 'y0': fit_0_average, 'y1': fit_03_average, 'y2': fit_06_average, 'y3': fit_085_average })
    plt.plot( 'x', 'y0', data=df, marker='', color='olive', linewidth=2, label="0")
    plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2, label="0.3")
    plt.plot( 'x', 'y2', data=df, marker='', color='orange', linewidth=2, label="0.6")
    plt.plot( 'x', 'y3', data=df, marker='', color='green', linewidth=2, label="0.85")
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()

#crossover_impact()
"""
Analysis of the impact of the mutation probability (1 point)

The impact of the mutation probability on the results should be investigated. For this
purpose, a minimum of 3 different mutation probability values must be selected and the
test for each value must be carried out at least 5 times. The averaged values of the best
individuals in subsequent generations should be shown on a chart. The conclusions of
the study should be described in the report.
"""
def mutation_impact():
    def gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size):
        generate(n, w, s, output_file)
        task = read(input_file)
        pop = init_population(n, POP_SIZE)
        i = 0
        fitness = []
        while (i < ITERATIONS):
            print("Iteration: " + str(i+1))
            j = 0
            new_pop = []
            while(j < POP_SIZE):
                parent1 = tournament(pop, tournament_size)
                parent2 = tournament(pop, tournament_size)
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                mutate(child, MUTATION_RATE)
                new_pop.append(child)
                j += 1
            pop = new_pop
            best_ind = tournament(pop, POP_SIZE)
            best_ind_fit = evaluate(best_ind, task)
            print("Fitness: " + str(best_ind_fit))
            fitness.append(best_ind_fit)
            i += 1
        return fitness

    POP_SIZE = 100
    ITERATIONS = 20
    CROSSOVER_RATE = 0.6
    MUTATION_RATE = 0
    tournament_size = 30
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_0_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    MUTATION_RATE = 0.01
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_001_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    MUTATION_RATE = 0.005
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_0005_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    MUTATION_RATE = 0.003
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_0003_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    df=pd.DataFrame({'x': range(ITERATIONS), 'y0': fit_0_average, 'y1': fit_001_average, 'y2': fit_0005_average, 'y3': fit_0003_average })
    plt.plot( 'x', 'y0', data=df, marker='', color='olive', linewidth=2, label="0")
    plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2, label="0.01")
    plt.plot( 'x', 'y2', data=df, marker='', color='orange', linewidth=2, label="0.005")
    plt.plot( 'x', 'y3', data=df, marker='', color='green', linewidth=2, label="0.003")
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()

#mutation_impact()

"""
Analysis of the impact of tournament size (1 point)

The impact of the tournament size on the results should be investigated. For this
purpose, a minimum of 3 different tournament size values must be selected and the test
for each value must be carried out at least 5 times. The averaged values of the best
individuals in subsequent generations should be shown on a chart. The conclusions of
the study should be described in the report
"""
def tournament_impact():
    def gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size):
        generate(n, w, s, output_file)
        task = read(input_file)
        pop = init_population(n, POP_SIZE)
        i = 0
        fitness = []
        while (i < ITERATIONS):
            print("Iteration: " + str(i+1))
            j = 0
            new_pop = []
            while(j < POP_SIZE):
                parent1 = tournament(pop, tournament_size)
                parent2 = tournament(pop, tournament_size)
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                mutate(child, MUTATION_RATE)
                new_pop.append(child)
                j += 1
            pop = new_pop
            best_ind = tournament(pop, POP_SIZE)
            best_ind_fit = evaluate(best_ind, task)
            print("Fitness: " + str(best_ind_fit))
            fitness.append(best_ind_fit)
            i += 1
        return fitness

    POP_SIZE = 100
    ITERATIONS = 20
    CROSSOVER_RATE = 0.6
    MUTATION_RATE = 0.005
    tournament_size = 10
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_10_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    tournament_size = 30
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_30_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    tournament_size = 50
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_50_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]


    df=pd.DataFrame({'x': range(ITERATIONS), 'y0': fit_10_average, 'y1': fit_30_average, 'y2': fit_50_average })
    plt.plot( 'x', 'y0', data=df, marker='', color='olive', linewidth=2, label="10")
    plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2, label="30")
    plt.plot( 'x', 'y2', data=df, marker='', color='orange', linewidth=2, label="50")
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()

#tournament_impact()

"""
Analysis of the impact of population size (1 point)

The impact of the population size on the results should be investigated. For this
purpose, a minimum of 3 different population size values must be selected and the test
for each value must be carried out at least 5 times. The averaged values of the best
individuals in subsequent generations should be shown on a chart. The conclusions of
the study should be described in the report.
"""
def pop_impact():
    def gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size):
        generate(n, w, s, output_file)
        task = read(input_file)
        pop = init_population(n, POP_SIZE)
        i = 0
        fitness = []
        while (i < ITERATIONS):
            print("Iteration: " + str(i+1))
            j = 0
            new_pop = []
            while(j < POP_SIZE):
                parent1 = tournament(pop, tournament_size)
                parent2 = tournament(pop, tournament_size)
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                mutate(child, MUTATION_RATE)
                new_pop.append(child)
                j += 1
            pop = new_pop
            best_ind = tournament(pop, POP_SIZE)
            best_ind_fit = evaluate(best_ind, task)
            print("Fitness: " + str(best_ind_fit))
            fitness.append(best_ind_fit)
            i += 1
        return fitness

    POP_SIZE = 100
    ITERATIONS = 20
    CROSSOVER_RATE = 0.6
    MUTATION_RATE = 0.005
    tournament_size = 50
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_100_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    POP_SIZE = 50
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_50_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]

    POP_SIZE = 200
    fit1 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit2 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit3 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit4 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit5 = gen_results(POP_SIZE, ITERATIONS, CROSSOVER_RATE, MUTATION_RATE, tournament_size)
    fit_200_average = [(fit1[j] + fit2[j] + fit3[j] + fit4[j] + fit5[j])/5 for j in range(ITERATIONS)]


    df=pd.DataFrame({'x': range(ITERATIONS), 'y0': fit_50_average, 'y1': fit_100_average, 'y2': fit_200_average })
    plt.plot( 'x', 'y0', data=df, marker='', color='olive', linewidth=2, label="50")
    plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2, label="100")
    plt.plot( 'x', 'y2', data=df, marker='', color='orange', linewidth=2, label="200")
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.show()

#pop_impact()

"""
Comparison of the best solution with any of the non-evolutionary methods (1 p.)

Compare the genetic algorithm (quality of the output, execution time) using the best set of parameters obtained from 
the previous evaluations with any of the non-evolutionary methods (but not a random method)
"""
def knapsack():
    generate(n, w, s, output_file)
    task = read(input_file)
    print("Weight: " + str(w))
    print("Size: " + str(s))
    print("n: " + str(n))
    sum_weight = 0
    sum_size = 0
    fitness = 0
    max_cost = 0
    task = task[1:]
    task = np.array(task,dtype=int)
    while ((sum_weight < w) and (sum_size < s)):
        max_cost = task.max(axis=0)[2]
        index = task.argmax(axis=0)[2]
        w_i = task[index][0]
        s_i = task[index][1]
        if(((sum_size + s_i) <= s) and ((sum_weight + w_i) <= w)):
            sum_weight += w_i
            sum_size += s_i
            fitness += max_cost
            task = np.delete(task, index, 0)
        else:
            break
    print("sum weight: " + str(sum_weight))
    print("sum size: " + str(sum_size))
    print(fitness)
    return fitness

#knapsack()
def knapsack_graph():
    k = []
    k_sum = 0
    tests = 1000
    for i in range(tests):
        k.append(knapsack())
        k_sum += k[i]
    mean_fitness = k_sum/tests
    df=pd.DataFrame({'x': range(1, tests+1), 'y': k })
    plt.plot( 'x', 'y', data=df, marker='o', linestyle="None")
    plt.xlabel("Tests")
    plt.ylabel("Fitness function")
    plt.show()

    print("Mean value of fitness function: " + str(mean_fitness))

#knapsack_graph()
