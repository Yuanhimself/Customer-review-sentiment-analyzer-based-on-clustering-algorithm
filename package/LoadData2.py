import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import threading
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import re
import jieba.analyse
from pyecharts.charts import WordCloud, Bar
from pyecharts import options as opts
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict

# 创建主窗口
root = tk.Tk()
root.title("商品评论情感分类")

# 添加文件选择函数
def select_file():
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

# 添加运行函数
def run_analysis():
    file_path = entry.get()
    if not file_path:
        messagebox.showerror("错误", "请选择数据文件")
        return

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("错误", f"读取文件失败：{e}")
        return

    data.dropna(axis=0, inplace=True)

    def remove_url(src):
        vTEXT = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】╮ ￣ ▽ ￣ ╭\\～⊙％；①（）：《》？“”‘’！[\\]^_`{|}~\s]+', "", src)
        return vTEXT

    cutted = []
    for row in data.values:
        text = remove_url(str(row[0]))
        raw_words = (' '.join(jieba.cut(text)))
        cutted.append(raw_words)

    cutted_array = np.array(cutted)

    data_cutted = pd.DataFrame({
        'Comment': cutted_array,
        'Class': data['Class']
    })

    with open("C:/Users/Lenovo/Desktop/Stopwords.txt", 'r', encoding='utf-8') as f:
        stopwords = [item.strip() for item in f]

    jieba.analyse.set_stop_words("C:/Users/Lenovo/Desktop/Stopwords.txt")

    vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=2000, min_df=6)
    data_transform = vectorizer.fit_transform(data_cutted['Comment'].values.tolist())

    X_train, X_test, y_train, y_test = train_test_split(data_transform, data_cutted['Class'], random_state=10, test_size=0.2)

    progress_bar.start()  # 启动进度条

    # 定义模型训练函数，以便在新线程中执行，避免界面卡顿
    def train_model():
        gnb = GaussianNB()
        gnb_model = gnb.fit(X_train.toarray(), y_train)
        gnb_prelabel = gnb_model.predict(X_test.toarray())

        svc = SVC(kernel='rbf', C=1)
        svc_model = svc.fit(X_train, y_train)
        svc_prelabel = svc_model.predict(X_test)

        # 显示模型报告
        messagebox.showinfo("结果", f"朴素贝叶斯模型报告：\n{classification_report(y_true=y_test, y_pred=gnb_prelabel)}\n\nSVM模型报告：\n{classification_report(y_true=y_test, y_pred=svc_prelabel)}")

        # 可视化部分
        visualize_data(data_cutted, stopwords)
        progress_bar.stop()  # 停止进度条

    # 创建新线程来执行模型训练函数
    t = threading.Thread(target=train_model)
    t.start()

def run_dbscan():
    file_path = entry.get()
    if not file_path:
        messagebox.showerror("错误", "请选择数据文件")
        return

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("错误", f"读取文件失败：{e}")
        return

    data.dropna(axis=0, inplace=True)

    def remove_url(src):
        vTEXT = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】╮ ￣ ▽ ￣ ╭\\～⊙％；①（）：《》？“”‘’！[\\]^_`{|}~\s]+', "", src)
        return vTEXT

    cutted = []
    for row in data.values:
        text = remove_url(str(row[0]))
        raw_words = (' '.join(jieba.cut(text)))
        cutted.append(raw_words)

    cutted_array = np.array(cutted)

    data_cutted = pd.DataFrame({
        'Comment': cutted_array,
        'Class': data['Class']
    })

    with open("C:/Users/Lenovo/Desktop/Stopwords.txt", 'r', encoding='utf-8') as f:
        stopwords = [item.strip() for item in f]

    jieba.analyse.set_stop_words("C:/Users/Lenovo/Desktop/Stopwords.txt")

    vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=2000, min_df=6)
    data_transform = vectorizer.fit_transform(data_cutted['Comment'].values.tolist())

    X_train, X_test, y_train, y_test = train_test_split(data_transform, data_cutted['Class'], random_state=10, test_size=0.2)

    progress_bar.start()  # 启动进度条

    # 定义DBSCAN算法训练函数
    def train_dbscan():
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(data_transform)

        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        messagebox.showinfo("结果", f"DBSCAN算法发现 {n_clusters_} 个情感簇")

        progress_bar.stop()  # 停止进度条

    # 创建新线程来执行DBSCAN算法训练函数
    t = threading.Thread(target=train_dbscan)
    t.start()


# 添加可视化函数
def visualize_data(data, stopwords):
    # 绘制好评和差评的词云图
    keywords_pos = jieba.analyse.extract_tags(''.join(data['Comment'][data['Class'] == 1]), withWeight=True, topK=30)
    keywords_neg = jieba.analyse.extract_tags('.join(data['Comment'][data['Class'] == -1]), withWeight=True, topK=30)

    wordcloud_pos = (
        WordCloud()
        .add(series_name="", data_pair=keywords_pos[:], word_size_range=[10, 66])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="好评关键词词云图", title_textstyle_opts=opts.TextStyleOpts(font_size=23)),
            tooltip_opts=opts.TooltipOpts(is_show=True))
    )

    wordcloud_neg = (
        WordCloud()
        .add(series_name="", data_pair=keywords_neg[:], word_size_range=[10, 66])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="差评关键词词云图", title_textstyle_opts=opts.TextStyleOpts(font_size=23)),
            tooltip_opts=opts.TooltipOpts(is_show=True))
    )

    # 绘制好评、中评、差评数量柱状图
    class_counts = data['Class'].value_counts()
    x_label = ['positive', 'negative', 'neutral']
    class_num = (
        Bar()
        .add_xaxis(x_label)
        .add_yaxis("", class_counts.to_list(), color=['#4c8dae'])
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar chart of the number of positive, neutral and negative reviews"))
    )

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(data_transform[:, 0], data_transform[:, 1], c=data['Class'], cmap=plt.cm.Set1, edgecolor='k', s=40)
    plt.title("Scatter Plot of Comments")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

    # 绘制饼状图
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=x_label, autopct='%1.1f%%', startangle=140)
    plt.title("Pie Chart of the Distribution of Sentiment Classes")
    plt.axis('equal')
    plt.show()

    # 展示词云图和柱状图
    wordcloud_pos.render("wordcloud_pos.html")
    wordcloud_neg.render("wordcloud_neg.html")
    class_num.render("class_num.html")

    # 显示柱状图
    plt.bar(x_label, class_counts)
    plt.title("Bar chart of the number of positive, neutral and negative reviews")
    plt.xlabel("Class")
    plt.ylabel("Number")
    plt.show()


# 添加文件选择按钮和输入框
label = tk.Label(root, text="请选择数据文件：")
label.grid(row=0, column=0, padx=10, pady=10)

entry = tk.Entry(root, width=50)
entry.grid(row=0, column=1, padx=10, pady=10)

button_select = tk.Button(root, text="选择文件", command=select_file)
button_select.grid(row=0, column=2, padx=10, pady=10)

# 添加运行按钮
button_run = tk.Button(root, text="运行分析", command=run_analysis)
button_run.grid(row=1, column=1, padx=10, pady=10)
button_run_dbscan = tk.Button(root, text="运行DBSCAN", command=run_dbscan)
button_run_dbscan.grid(row=1, column=2, padx=10, pady=10)

# 添加进度条
progress_bar = Progressbar(root, orient=tk.HORIZONTAL, mode='indeterminate', length=200)
progress_bar.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()

