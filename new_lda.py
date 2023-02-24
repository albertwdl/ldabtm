import pandas as pd
from gensim import corpora, models

# 读取csv文件
df = pd.read_csv("lad.csv")

# 提取“reviewinfo”列中的文本数据，并将所有数据类型转换为字符串类型
texts = [str(document) for document in df['reviewinfo'].tolist()]

# 建立文本语料库
texts = [[word for word in document.lower().split()] for document in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 训练LDA模型
num_topics = 10  # 设置主题数量
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

# 保存每个主题的top 10 topic words到文件
topic_words = []
for i, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
    top_words = [word[0] for word in topic[:10]]
    topic_words.append(top_words)
    with open("top-topic-words.csv", "a") as f:
        f.write(f"Topic {i}: {', '.join(top_words)}\n")

# 保存主题概率分布到文件
distributions = []
for i in range(num_topics):
    prob_distribution = lda_model.get_topic_terms(i, topn=10)
    distribution = [round(prob[1], 2) for prob in prob_distribution]
    distributions.append(distribution)
    with open("distribution.csv", "a") as f:
        f.write(f"Topic {i}: {', '.join(map(str, distribution))}\n")

# 打印结果
print("Top 10 topic words for each topic are saved in 'top-topic-words.csv'.")
print("Topic probability distributions are saved in 'distribution.csv'.")