import matplotlib.pyplot as plt
import re

plt.rc('font', size=20)        # 기본 폰트 크기
#plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
#plt.rc('xtick', labelsize=50)  # x축 눈금 폰트 크기
#plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
#plt.rc('legend', fontsize=12)  # 범례 폰트 크기
#plt.rc('figure', titlesize=50) # figure title 폰트 크기

def parse_file(file_path, fixed_input_token):
    output_tokens = []
    tokens_per_s = []
    hit_rates = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'input_token:\s*(\d+),\s*output_token:\s*(\d+),.*hit_rate:\s*([\d\.]+),([\d\.]+)token/s', line)
            if match:
                input_token = int(match.group(1))
                if input_token == fixed_input_token:
                    output_tokens.append(int(match.group(2)))
                    hit_rates.append(float(match.group(3)))
                    tokens_per_s.append(float(match.group(4)))

    return output_tokens, tokens_per_s, hit_rates

def plot_comparison(output_tokens1, tokens_per_s1, hit_rates1, output_tokens2, tokens_per_s2, hit_rates2, fixed_input_token, output_file):
    fig, ax1 = plt.subplots()

    # file1의 throughput (token/s)은 빨간색 실선으로 표시
    ax1.set_xlabel(f'Output Token (Fixed Input Token = {fixed_input_token})')
    ax1.set_ylabel('Throughput (token/s)')#, color='tab:red')
    ax1.plot(output_tokens1, tokens_per_s1, color='red', label='Fiddlestick Token/s', linestyle='-')
    
    # file2의 throughput (token/s)은 파란색 실선으로 표시
    ax1.plot(output_tokens2, tokens_per_s2, color='blue', label='Fiddler Token/s', linestyle='-')
    ax1.tick_params(axis='y')#, labelcolor='tab:red')
    all_output_tokens = sorted(set(output_tokens1 + output_tokens2))
    ax1.set_xticks(all_output_tokens)

    # x축과 첫 번째 y축이 0부터 시작하도록 설정
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0,top=0.5)
    ax1.grid()

    ax2 = ax1.twinx()

    # file1의 hit rate은 빨간색 점선으로 표시
    ax2.set_ylabel('Hit Rate')#, color='tab:blue')
    ax2.plot(output_tokens1, hit_rates1, color='red', label='Fiddlestick Hit Rate', linestyle='--')
    
    # file2의 hit rate은 파란색 점선으로 표시
    ax2.plot(output_tokens2, hit_rates2, color='blue', label='Fiddler Hit Rate', linestyle='--')
    ax2.tick_params(axis='y')#, labelcolor='tab:blue')

    # 두 번째 y축이 0부터 시작하도록 설정
    ax2.set_ylim(bottom=0,top=0.10)

    # 범례 표시
    #fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.05), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")

# 파일 경로와 고정된 input_token 값을 설정
file_path1 = '0317.txt'
file_path2 = '0317_fiddler.txt'
fixed_input_token = 64

# 두 파일의 데이터를 파싱
output_tokens1, tokens_per_s1, hit_rates1 = parse_file(file_path1, fixed_input_token)
output_tokens2, tokens_per_s2, hit_rates2 = parse_file(file_path2, fixed_input_token)

# 데이터를 비교하여 플롯하고 파일로 저장
output_file = '0626_input_64.png'
plot_comparison(output_tokens1, tokens_per_s1, hit_rates1, output_tokens2, tokens_per_s2, hit_rates2, fixed_input_token, output_file)


