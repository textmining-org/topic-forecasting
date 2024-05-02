# from collections import Counter
# import matplotlib.pyplot as plt
#
# # 'text' 컬럼의 모든 단어를 하나의 리스트로 합칩니다.
# import pandas as pd
#
# df1 = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/results/patents_201701_202312.csv')
#
# text_columns = df1['text']
#
# # 단어의 빈도수를 계산합니다.
# word_counts = Counter(text_columns)
#
# # 가장 흔히 등장하는 30개 단어와 그 빈도수를 추출합니다.
# most_common_words = word_counts.most_common(30)
#
# # 단어와 빈도수를 분리하여 리스트로 만듭니다.
# words, frequencies = zip(*most_common_words)
#
# # Zipf의 그래프를 그립니다.
# plt.figure(figsize=(10, 8))
# plt.barh(range(len(words)), frequencies, tick_label=words)
# plt.gca().invert_yaxis() # 높은 빈도수를 가진 단어를 위로 표시하기 위해 y축을 뒤집습니다.
# plt.xlabel('Frequency')
# plt.title('Top 30 Most Common Words')
# plt.show()

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# CSV 파일 로드
df1 = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining-main/team_project/_datasets/preprocess/results/patent_201701_202312_with_duplicates.csv')

# 'text' 컬럼의 모든 행에서 리스트 내의 모든 단어들을 하나의 리스트로 합칩니다.
combined_words = []
for text in df1['text']:
    # 각 행의 'text' 컬럼이 문자열 형태의 리스트이므로, 이를 실제 리스트로 변환합니다.
    # 안전을 위해 eval 사용을 피하고, JSON 파싱을 사용합니다.
    import json
    try:
        words_list = json.loads(text.replace("'", '"'))
    except json.decoder.JSONDecodeError:
        # JSON으로 변환할 수 없는 경우, 빈 리스트를 사용
        words_list = []
    combined_words.extend(words_list)

# 단어의 빈도수를 계산합니다.
word_counts = Counter(combined_words)

# 가장 흔히 등장하는 30개 단어와 그 빈도수를 추출합니다.
most_common_words = word_counts.most_common(50)

# 단어와 빈도수를 분리하여 리스트로 만듭니다.
words, frequencies = zip(*most_common_words)

# Zipf의 그래프를 그립니다.
plt.figure(figsize=(10, 8))
plt.barh(range(len(words)), frequencies, tick_label=words)
plt.gca().invert_yaxis() # 높은 빈도수를 가진 단어를 위로 표시하기 위해 y축을 뒤집습니다.
plt.xlabel('Frequency')
plt.title('Top 30 Most Common Words')
plt.show()
