import matplotlib.pyplot as plt

plt.rc('font', size=15)        # 기본 폰트 크기
#plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
#plt.rc('xtick', labelsize=50)  # x축 눈금 폰트 크기
#plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
#plt.rc('legend', fontsize=12)  # 범례 폰트 크기
#plt.rc('figure', titlesize=50) # figure title 폰트 크기


# 데이터
x = [16, 32, 64, 128]
Fiddlestick = [6.37, 5.56, 2.79, 1.90]
Fiddler = [6.87, 8.57, 10.89, 10.56]

# 그래프 그리기
plt.plot(x, Fiddlestick, 'r-', marker='o', markersize=3, label='Fiddlestick')
plt.plot(x, Fiddler, 'b-', marker='o', markersize=3, label='Fiddler')

# 각 점에 값 표시
#for i, val in enumerate(Fiddlestick):
#    plt.text(x[i], val + 0.3, f"{val:.2f}", color='red', ha='center')

#for i, val in enumerate(Fiddler):
#    plt.text(x[i], val + 0.3, f"{val:.2f}", color='blue', ha='center')

for i, val in enumerate(Fiddlestick):
    offset = -0.6 if i == 0 else 0.3
    plt.text(x[i], val + offset, f"{val:.2f}", color='red', ha='center')

# Fiddler: 라벨을 기본보다 살짝 위로
for i, val in enumerate(Fiddler):
    offset = 0.5 if i == 0 else 0.3
    plt.text(x[i], val + offset, f"{val:.2f}", color='blue', ha='center')

# 레이블 및 제목
plt.xlabel("Input token")
plt.ylabel("Latency (s/token)")
plt.ylim(top=max(max(Fiddlestick), max(Fiddler)) + 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 그래프 저장
plt.savefig("latency_plot.png", dpi=300)  # dpi를 높이면 해상도가 올라갑니다

# 화면에 띄우지 않음
# plt.show()

