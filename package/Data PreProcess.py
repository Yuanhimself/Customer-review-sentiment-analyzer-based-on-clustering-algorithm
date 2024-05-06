import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas
# 缺失值分析
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
axs = np.ravel(ax)
msno.matrix(df, fontsize=9, color=(0.25, 0, 0.5), ax=axs[0]);
msno.bar(df, fontsize=8, color=(0.25, 0, 0.5), ax=axs[1]);
msno.heatmap(df, fontsize=8, ax=axs[2]);
msno.dendrogram(df, fontsize=8, ax=axs[3], orientation='top')

fig.suptitle('Missing Values Analysis', y=1.01, fontsize=15)
# plt.savefig('missing_values_analysis.png') # 保存图片
plt.show()
