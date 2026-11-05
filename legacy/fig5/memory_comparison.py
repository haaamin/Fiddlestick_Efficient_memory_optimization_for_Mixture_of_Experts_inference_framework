import matplotlib.pyplot as plt
import numpy as np


plt.rc('font', size=18)
# 데이터
models = ['Fiddlestick', 'Fiddler']
memory_requirements = [1.24, 4.15]
colors = ['red', 'blue']

# y 위치 직접 지정 (간격 좁힘)
y_pos = [0, 0.4]  # 두 막대 사이 간격을 기본보다 줄임

# 그래프 크기
plt.figure(figsize=(8, 2.5))

# 막대 그리기 (수동 y 위치)
bars = plt.barh(y_pos, memory_requirements, color=colors, height=0.2)

# 모델 이름 수동 표시
plt.yticks(y_pos, models)

# 수치 표시
for bar, mem in zip(bars, memory_requirements):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f"{mem:.2f} GB", va='center', color='black')

# x축 설정
plt.xlabel("Memory Requirement (GB)")
plt.xticks(np.arange(0, 5.5, 0.5))
plt.grid(axis='x', linestyle='--', linewidth=0.5)

# 저장 및 출력
plt.tight_layout()
plt.savefig("memory_usage_tighter_gap.png", dpi=300)
#plt.show()

