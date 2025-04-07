import matplotlib.pyplot as plt
import pandas as pd

news_df = pd.read_csv("./news_07.tsv", sep="\t")
papers_df = pd.read_csv("./papers_10.tsv", sep="\t")
patents_df = pd.read_csv("./patents_13.tsv", sep="\t")

# Year_Month 컬럼 추출 및 변환
for df in [news_df, papers_df, patents_df]:
    df['Year_Month'] = df['Unnamed: 0'].apply(lambda x: x.split(":")[1])
    df['Year'] = df['Year_Month'].apply(lambda x: int(x.split("_")[0]))
    df['Month'] = df['Year_Month'].apply(lambda x: int(x.split("_")[1]))

def calculate_total_word_count(df):
    df['Total_Word_Count'] = df.drop(['Year_Month', 'Year', 'Month'], axis=1).sum(axis=1)
    # df['Word_Ratio'] = df['Total_Word_Count']
    return df[['Year_Month', 'Total_Word_Count', 'Year', 'Month']]

# 각 데이터프레임에 대해 word ratio 계산
news_ratio = calculate_total_word_count(news_df)
papers_ratio = calculate_total_word_count(papers_df)
patents_ratio = calculate_total_word_count(patents_df)
print(patents_ratio)

# Year_Month을 문자열로 변환 후 정렬
news_ratio['Year_Month'] = news_ratio['Year_Month'].astype(str)
papers_ratio['Year_Month'] = papers_ratio['Year_Month'].astype(str)
patents_ratio['Year_Month'] = patents_ratio['Year_Month'].astype(str)

# 각 데이터프레임 정렬
news_ratio = news_ratio.sort_values('Year_Month')
papers_ratio = papers_ratio.sort_values('Year_Month')
patents_ratio = patents_ratio.sort_values('Year_Month')

# x축 값 전체를 설정하고, x축 값이 포함되는지 확인
all_months = sorted(set(news_ratio['Year_Month']).union(set(papers_ratio['Year_Month'])).union(set(patents_ratio['Year_Month'])))

# 2, 4, 6, 8, 10, 12 월만 표시하기 위해 x축 레이블 필터링
filtered_months = [month for month in all_months if month.split("_")[1] in ['02', '04', '06', '08', '10', '12']]

# 플롯 크기 설정: width를 더 늘려서 간격 확보
fig, ax1 = plt.subplots(figsize=(40, 25))  # 기존보다 가로 크기를 더 크게 설정하여 레이블 간의 간격 확보

# 첫 번째 y축에 데이터 플롯 생성 (뉴스와 논문)
ax1.plot(news_ratio['Year_Month'], news_ratio['Total_Word_Count'], color='orange', marker='o', label='News (Topic 7)')
ax1.plot(papers_ratio['Year_Month'], papers_ratio['Total_Word_Count'], color='red', marker='o', label='Papers (Topic 10)')
ax1.set_xlabel('Month', fontsize=60, labelpad=10)
ax1.set_ylabel('Total Word Count (News & Papers)', fontsize=60)
ax1.tick_params(axis='y', labelsize=32)
ax1.tick_params(axis='x', labelsize=32, rotation=70)
ax1.set_xticks(filtered_months)
ax1.set_xlim([all_months[0], '2023_12'])

# 두 번째 y축 생성 (특허)
ax2 = ax1.twinx()
ax2.plot(patents_ratio['Year_Month'], patents_ratio['Total_Word_Count'], color='blue', marker='o', label='Patents (Topic 13)')
ax2.set_ylabel('Total Word Count (Patents)', fontsize=60)
ax2.tick_params(axis='y', labelsize=32)

# 범례 설정
fig.legend(fontsize=60, markerscale=3, prop={'size': 54}, loc='upper left', bbox_to_anchor=(0.1, 1))

# x축 레이블과 축 간격 조정
plt.gca().tick_params(axis='x', which='major', pad=15)  # pad 값으로 레이블과 축 사이 간격 설정

# x축 레이블 간 간격 조정 및 y축 상단 여백 설정
plt.subplots_adjust(top=0.85, bottom=0.3)  # 아래쪽 여백(bottom)을 더 늘려 레이블이 잘 보이도록 설정

# 레이아웃 조정 및 이미지 저장
plt.tight_layout()
plt.savefig('trend_analysis_plot_filtered_months_spacing_dual_yaxis.png', dpi=300)  # 이미지 저장
plt.show()
