import numpy as np
import pandas as pd
import jieba
import pyLDAvis
from biterm.btm import oBTM 
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions
import re

# 待做 LDA 的文本 csv 文件，可以是本地文件，也可以是远程文件，一定要保证它是存在的！！！！
# source_csv_path = 'answers.csv'
source_csv_path = './lad.csv'
# 文本 csv 文件里面文本所处的列名,注意这里一定要填对，要不然会报错的！！！
# document_column_name = '回答内容'
document_column_name = 'reviewinfo'
# 输出主题词的文件路径
top_words_csv_path = 'btm-top-topic-words.txt'
# 输出各文档所属主题的文件路径
predict_topic_csv_path = 'btm-document-distribution.txt'
# 可视化 html 文件路径
html_path = 'btm-document-visualization.html'
# 选定的主题数
n_topics = 3
# 迭代次数
n_iters = 2
# 要输出的每个主题的前 n_top_words 个主题词数
n_top_words = 20
# 去除无意义字符的正则表达式
pattern = u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'
# 去除英文和数字的正则表达式
eng_num_pattern = u'[0-9a-zA-Z]+'

df = (
    pd.read_csv(
        source_csv_path,
        encoding='utf-8-sig')
    .drop_duplicates()
    .rename(columns={
        document_column_name: 'text'
    }))
# 设置停用词集合
stop_words_set = set(['你', '我'])
# 去重、去缺失、分词、去英文
df['cut'] = (
    df['text']
    .apply(lambda x: str(x))
    .apply(lambda x: re.sub(eng_num_pattern, ' ', x))
    .apply(lambda x: re.sub(pattern, ' ', x))
    .apply(lambda x: " ".join([word for word in jieba.lcut(x) if word not in stop_words_set]))
)

texts = list(df['cut'].values)

# vectorize texts
stop_words_list = ['你', '我']
vec = CountVectorizer(stop_words=stop_words_list, dtype=np.float32)
X = vec.fit_transform(texts).toarray()

# get vocabulary
vocab = np.array(vec.get_feature_names())

# get biterms
biterms = vec_to_biterms(X)

# create btm
btm = oBTM(num_topics=n_topics, V=vocab)

print("\n\n Train Online BTM ..")
for i in range(0, len(biterms), 1000): # prozess chunk of 200 texts
    biterms_chunk = biterms[i:i + 1000]
    btm.fit(biterms_chunk, iterations=n_iters)
topics = btm.transform(biterms)

print("\n\n Visualize Topics ..")
vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
pyLDAvis.save_html(vis, html_path)

print("\n\n Topic coherence ..")
res = topic_summuary(btm.phi_wz.T, X, vocab, 10)
with open(file=top_words_csv_path, mode='w+') as file:
    for z in range(len(res['coherence'])):
        file.writelines('Topic {} | Coherence={:0.2f} | Top words= {}\n'.format(z, res['coherence'][z], ' '.join(res['top_words'][z])))

print("\n\n Texts & Topics ..")
with open(file=predict_topic_csv_path, mode='w+') as file:
    for i in range(len(texts)):
        file.writelines("(topic: {}) {}\n".format(topics[i].argmax(), texts[i]))