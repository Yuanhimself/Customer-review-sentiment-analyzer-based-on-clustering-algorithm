# 载入必要库
import jieba
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt 
import pyecharts.options as opts
from pyecharts.charts import WordCloud
from pyecharts.charts import Bar
import re
#logging
import warnings
warnings.filterwarnings('ignore')

#读入数据集
data = pd.read_csv("C:/Users/Lenovo/Desktop/中文商品评论.csv")
data.head(10)

# 数据集的大小

data.info()

# 移除含有缺失值的行
data.dropna(axis=0,inplace=True)
data.shape

def remove_url(src):
    # 去除标点符号、数字、字母
    vTEXT = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】╮ ￣ ▽ ￣ ╭\\～⊙％；①（）：《》？“”‘’！[\\]^_`{|}~\s]+', "", src)
    return vTEXT

cutted = []
for row in data.values:
        text = remove_url(str(row[0])) #去除文本中的标点符号、数字、字母
        raw_words = (' '.join(jieba.cut(text)))#分词,并用空格进行分隔
        cutted.append(raw_words)

cutted_array = np.array(cutted)

# 生成新数据文件，Comment字段为分词后的内容
data_cutted = pd.DataFrame({
    'Comment': cutted_array,
    'Class': data['Class']
})
data_cutted.head()#查看分词后的数据集

with open("C:/Users/Lenovo/Desktop/Stopwords.txt", 'r', encoding='utf-8') as f:  # 读停用词表
    stopwords = [item.strip() for item in f]  # 通过列表推导式的方式获取所有停用词

for i in stopwords[:100]:  # 读前100个停用词
    print(i, end='')

#设定停用词文件,在统计关键词的时候，过滤停用词
import jieba.analyse
jieba.analyse.set_stop_words("C:/Users/Lenovo/Desktop/Stopwords.txt")
data_cutted['Comment'][data_cutted['Class'] == 1]

# 好评关键词
keywords_pos = jieba.analyse.extract_tags(''.join(data_cutted['Comment'][data_cutted['Class'] == 1]),withWeight = True,topK=30)
for item in keywords_pos:
    print(item[0],end=' ')

#差评关键词
keywords_neg = jieba.analyse.extract_tags(''.join(data_cutted['Comment'][data_cutted['Class'] == -1]),withWeight = True,topK=30)

for item in keywords_neg:
    print (item[0],end=' ')

data_cutted['Class'].value_counts()

# 不同类别数据记录的统计
x_label = ['好评','差评','中评']
class_num = (
    Bar()
    #设置x轴的值
    .add_xaxis(x_label)
    #设置y轴数据
    .add_yaxis("",data_cutted['Class'].value_counts().to_list(),color=['#4c8dae'])
    #设置title
    .set_global_opts(title_opts=opts.TitleOpts(title="好评、中评、差评数量柱状图"))
)
class_num.render_notebook()

wordcloud_pos = (
        WordCloud()
        #data_pair：要绘制词云图的数据
        .add(series_name="", data_pair=keywords_pos[:], word_size_range=[10, 66])
        .set_global_opts(
            title_opts=opts.TitleOpts(
                #设置词云图标题和标题字号
                title="好评关键词词云图", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True))
)
wordcloud_pos.render_notebook()



wordcloud_neg = (
        WordCloud()
        #data_pair：要绘制词云图的数据
        .add(series_name="", data_pair=keywords_neg[:], word_size_range=[10, 66])
        .set_global_opts(
            title_opts=opts.TitleOpts(
                #设置词云图标题和标题字号
                title="差评关键词词云图", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True))
)
wordcloud_neg.render_notebook()

# 实现向量化方法
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = stopwords,max_df=2000,min_df=6)

#将文本向量化后的数据赋给data_transform
data_transform = vectorizer.fit_transform(data_cutted['Comment'].values.tolist())
#文本的词汇表
vectorizer.get_feature_names_out()
#调用toarray()方法查看文本向量化后的数据
data_transform.toarray()
data_transform.shape

#高斯朴素贝叶斯模型
from sklearn.model_selection import train_test_split #数据集划分

X_train, X_test, y_train, y_test = train_test_split(data_transform, data_cutted['Class'],
                                                   random_state=10,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb_model = gnb.fit(X_train.toarray(),y_train)
from sklearn.metrics import classification_report
gnb_prelabel = gnb_model.predict(X_test.toarray())

print(classification_report(y_true=y_test,y_pred=gnb_prelabel))

#SVM构建
from sklearn.svm import SVC

#设置kernel为‘rbf’高斯核，C=1
svc = SVC(kernel='rbf', C=1)
svc_model = svc.fit(X_train,y_train)
svc_prelabel = svc_model.predict(X_test)
print(classification_report(y_true=y_test,y_pred=svc_prelabel))