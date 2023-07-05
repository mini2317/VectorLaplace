import math
from collections import deque

WIDTH = 50
HEIGHT = 50
DX = 0.1         # 위치 간격
DT = 0.01        # 시간 간격
V = 1.0          # 파동의 속도

def posToVector(p1, p2, forceValue) -> tuple:
    '''
    위치를 벡터(튜플)로 바꿔준다. 마우스를 클릭했을 때 어느정도의 힘을 줄 지 계산할 수 있다.
    '''
    theta = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return (forceValue * math.cos(theta), forceValue * math.sin(theta))

class VectorField:
    '''
    벡터장이다. 라플라스 방정식의 원리를 이용해서 각 점에서의 힘을 계산할 수 있다.
    '''
    def __init__(self, width = WIDTH, height = HEIGHT) -> None:
        '''
        VectorField(width = WIDTH, height = HEIGHT)
        기본 값으로 WIDTH와 HEIGHT를 제공한다. 가로 width, 세로 height 크기의 벡터장을 만든다.
        '''
        self.width = width
        self.height = height
        self.field = [[(0, 0)] * width for _ in range(height)]
        self.visited = [[False] * width for _ in range(height)]
    
    def gaussian(self, x, dx = DX):
        return math.exp(-((x - 0.5 * self.width * dx) / (0.1 * self.width * dx))**2)

    def addForce(self, pos, force, team = None) -> None:
        '''
        addForce(pos, force) -> None
        pos는 1개 이상의 좌표들이 담긴 리스트로 주어져야 한다. 
        force는 1개 이상의 벡터들이 담긴 리스트로 주어져야 한다.
        '''
        startPoints = deque()
        for i in range(len(pos)):
            np = self.field[pos[i][0]][pos[i][1]]
            self.field[pos[i][0]][pos[i][1]] = (np[0] + force[i][0], np[1] + force[i][1])
            self.visited[pos[i][0]][pos[i][1]] = True
            startPoints.append(pos[i])
        self.getPointForce(startPoints)

    def getPointForce(self, startPoints: deque, dx = DX, dt = DT, v = V) -> None:
        '''
        BFS를 이용해서 각 점에서의 힘을 구할 수 있다.
        '''
        queue = deque()
        while startPoints:
            y, x = startPoints.popleft()
            if x != 0:
                if not self.visited[y][x - 1]:
                    self.visited[y][x - 1] = True
                    queue.append((y, x - 1))
            if x != self.width- 1:
                if not self.visited[y][x + 1]:
                    self.visited[y][x + 1] = True
                    queue.append((y, x + 1))
            if y != 0:
                if not self.visited[y - 1][x]:
                    self.visited[y - 1][x] = True
                    queue.append((y - 1, x))
            if y != self.height - 1:
                if not self.visited[y + 1][x]:
                    self.visited[y + 1][x] = True
                    queue.append((y + 1, x))
        while queue:
            y, x = queue.popleft()
            vectors = []
            if x != 0:
                vectors.append(self.field[y][x - 1])
                if not self.visited[y][x - 1]:
                    self.visited[y][x - 1] = True
                    queue.append((y, x - 1))
            if x != self.width- 1:
                vectors.append(self.field[y][x + 1])
                if not self.visited[y][x + 1]:
                    self.visited[y][x + 1] = True
                    queue.append((y, x + 1))
            if y != 0:
                vectors.append(self.field[y - 1][x])
                if not self.visited[y - 1][x]:
                    self.visited[y - 1][x] = True
                    queue.append((y - 1, x))
            if y != self.height - 1:
                vectors.append(self.field[y + 1][x])
                if not self.visited[y + 1][x]:
                    self.visited[y + 1][x] = True
                    queue.append((y + 1, x))
            np = self.field[y][x]
            vectors.append(np)
            psix = self.gaussian(self.field[y][x][0], dx = dx), self.gaussian(self.field[y][x + 1][0], dx = dx) if x < self.width - 1 else 0, self.gaussian(self.field[y][x - 1][0], dx = dx) if x > 0 else 0
            psiy = self.gaussian(self.field[y][x][0], dx = dx), self.gaussian(self.field[y - 1][x][0], dx = dx) if y < self.height - 1 else 0, self.gaussian(self.field[y - 1][x][0], dx = dx) if y > 0 else 0
            xg = 2 * psix[0] - psix[-1] - psix[1] + (v*dt/dx)**2 * (psix[-1] - 2 * psix[0] + psix[1]) - psiy[-1] - psiy[1] + (v*dt/dx)**2 * (psiy[-1] - 2*psiy[0] + psiy[1])
            psix = self.gaussian(self.field[y][x][1], dx = dx), self.gaussian(self.field[y][x + 1][1], dx = dx) if x < self.width - 1 else 0, self.gaussian(self.field[y][x - 1][1], dx = dx) if x > 0 else 0
            psiy = self.gaussian(self.field[y][x][1], dx = dx), self.gaussian(self.field[y - 1][x][1], dx = dx) if y < self.height - 1 else 0, self.gaussian(self.field[y - 1][x][1], dx = dx) if y > 0 else 0
            yg = 2 * psix[0] - psix[-1] - psix[1] + (v*dt/dx)**2 * (psix[-1] - 2 * psix[0] + psix[1]) - psiy[-1] - psiy[1] + (v*dt/dx)**2 * (psiy[-1] - 2*psiy[0] + psiy[1])
            self.field[y][x] = (yg, xg)
        self.visited = [[False] * self.width for _ in range(self.height)]

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

sum_ = lambda x : x*(x+1)//2
gf = GameField()
gf.force.addForce([(HEIGHT//2-1, WIDTH//2-1)],[(10,5)])
gf.update()
import matplotlib.pyplot as plt
data = gf.force.field
plt.figure(figsize=(WIDTH,HEIGHT))
plt.imshow([list(map(lambda x : int((math.sqrt(x[0]*x[0] + x[1]*x[1]))), data[i])) for i in range(HEIGHT)])
plt.show()
plt.figure(figsize=(WIDTH,HEIGHT))
plt.imshow(gf.team)
print(gf.team)
plt.show()
for i in range(HEIGHT) : plt.plot(list(map(lambda x : ((math.sqrt(x[0]*x[0] + x[1]*x[1]))), data[i])))
plt.show()