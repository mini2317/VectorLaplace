import math, decimal, copy
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 50
HEIGHT = 50
HORIZONTAL = 0
VERTICAL = 1
DT = 0.01
ALPHA = 0.1
ALPHA /= 4

def posToVector(p1, p2, forceValue) -> tuple:
    theta = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return (forceValue * math.cos(theta), forceValue * math.sin(theta))

class VectorField:
    def __init__(self, width = WIDTH, height = HEIGHT) -> None:
        self.width = width
        self.height = height
        self.field = np.zeros((width, height, 2))
    
    def addForce(self, pos, force, team = None) -> None:
        for i in range(len(pos)):
            nv = self.field[pos[i][1], pos[i][0]]
            self.field[pos[i][1], pos[i][0]] = (nv[1] + force[i][0], nv[0] + force[i][1])
        self.getPointForce()

    def getPointForce(self, alpha = ALPHA, dt = DT) -> None:
        newField = np.zeros((self.width, self.height, 2))
        for _ in range(int(1/dt)):
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    newField[x, y, 0] = alpha * (self.field[x + 1, y, 0] + self.field[x - 1, y, 0] + self.field[x, y + 1, 0] + self.field[x, y - 1, 0] - 4 * self.field[x, y, 0]) + self.field[x, y, 0]
                    newField[x, y, 1] = alpha * (self.field[x + 1, y, 1] + self.field[x - 1, y, 1] + self.field[x, y + 1, 1] + self.field[x, y - 1, 1] - 4 * self.field[x, y, 1]) + self.field[x, y, 1]
            self.field = newField[:, :, :]
            if _%10 == 0:
                for i in range(HEIGHT) : plt.plot(list(map(lambda x : ((math.sqrt(x[0]*x[0] + x[1]*x[1]))), self.field[i])))
                plt.show()

class GameField:
    def __init__(self, width = WIDTH, height = HEIGHT) -> None:
        self.force = VectorField(width, height)
        self.width, self.height = width, height
        self.team = [[0] * width for _ in range(height)]

    def update(self) :
        for y in range(self.width):
            for x in range(self.height):
                nv = self.force.field[y][x]
                if nv[0] ** 2 + nv[1] ** 2 >= 1:
                    nv = (int(nv[0]), int(nv[1]))
                    g = max(1, math.gcd(*nv))
                    n = (nv[0]//g, nv[1]//g)
                    for i in range(g):
                        if self.height > y + i * n[0] > 0 and self.width > x + i * n[1] > 0:
                            self.team[y + i * n[0]][x + i * n[1]] = 1

gf = GameField()
midWidth = WIDTH//2-1
midHeight = HEIGHT//2-1
gf.force.addForce([(midWidth, midHeight)],[(100,10)])
gf.update()

data = gf.force.field
plt.figure(figsize=(WIDTH,HEIGHT))
plt.imshow([list(map(lambda x : ((math.sqrt(x[0]*x[0] + x[1]*x[1]))), data[i])) for i in range(HEIGHT)])
plt.show()
plt.figure(figsize=(WIDTH,HEIGHT))
plt.imshow(gf.team)
plt.show()
for i in range(HEIGHT) : plt.plot(list(map(lambda x : ((math.sqrt(x[0]*x[0] + x[1]*x[1]))), data[i])))
plt.show()