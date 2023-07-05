import numpy as np
import matplotlib.pyplot as plt

# 파동방정식을 계산할 매개 변수 설정
num_points = 100  # 위치(x)를 이산화할 개수
num_steps = 200   # 시간(t)을 이산화할 단계
dx = 0.1         # 위치 간격
dt = 0.01        # 시간 간격
v = 1.0          # 파동의 속도

# 초기 조건 설정
x = np.linspace(0, num_points*dx, num_points)
psi = np.exp(-((x - 0.5*num_points*dx) / (0.1*num_points*dx))**2)  # 초기 파동함수

# 파동방정식 계산
for step in range(num_steps):
    psi_new = np.zeros(num_points)
    for i in range(1, num_points-1):
        psi_new[i] = 2*psi[i] - psi[i-1] - psi[i+1] + (v*dt/dx)**2 * (psi[i-1] - 2*psi[i] + psi[i+1])
    psi = psi_new

# 결과 그래프 출력
plt.plot(x, psi)
plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('Wave Equation Simulation')
plt.show()
