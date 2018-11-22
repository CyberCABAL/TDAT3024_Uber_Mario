#matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rand

def graphMarIO(name, delim): #Delim seperates the values in the file
    file = open(name, 'r')
    plots = np.asarray(file.read().split(delim),dtype='float')
    x_vals =[]
    for i in range(len(plots)):
        x_vals.append(i)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.plot(x_vals, plots, 'r-')
    plt.grid(True)
    plt.show()
    return

def graphMemory(name, delim):
    file = open(name, 'r')
    plots = np.asarray(file.read().split(delim), dtype='float')
    x_vals = []
    for i in range(len(plots)):
        x_vals.append(i)
    plt.plot(x_vals, plots)
    plt.grid(True)
    plt.show()


def test(rounds):
    color = ['c', 'r','k','b']
    vals1 = []
    vals2 = []
    for _ in range (rounds):
        vals1.append(_)
        vals2.append(rand.randint(0,500))
    plt.grid(True)
    plt.plot(vals1, vals2, 'r')
    plt.show()

def writeTest(inp):
    file = open('testfile.txt', 'w')
    file.write(str(rand.randint(0,300)))
    for _ in range(inp):
        file.write(","+str(rand.randint(0,300)))
    
#writeTest(200)
graphMarIO('testfile.txt', ',')
