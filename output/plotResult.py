import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ic = pd.read_csv('./result_HIST_all_ic.csv')
icdata = np.zeros(ic.shape[0])
for i in range(ic.shape[0]):
    icdata[i] = ic['all_ic'][i]
# print(data)

rankic = pd.read_csv('./result_HIST_all_rank_ic.csv')
rankicdata = np.zeros(rankic.shape[0])
for i in range(rankic.shape[0]):
    rankicdata[i] = rankic['all_rank_ic'][i]

# plt.ion()
plt.figure(1)
plt.rcParams.update({'font.size': 18})
plt.plot(range(ic.shape[0]),icdata,'o-',label="IC")
plt.plot(range(rankic.shape[0]),rankicdata,'o-',label="Rank IC")
plt.title('Score IC for HIST Model')
plt.xlabel('Times')
plt.legend()
# plt.show()


precision = pd.read_csv('./result_HIST_all_precision.csv')
precisiondata = np.zeros((8,precision.shape[0]))
col = ["1","3","5","10","20","30","50","100"]
plt.figure(2)
for j,c in enumerate(col):
    for i in range(precision.shape[0]):
        precisiondata[j,i] = precision[c][i]
    plt.plot(range(precision.shape[0]),precisiondata[j],'o-',label="precision_"+c)

plt.title('Score Precision for HIST Model')
plt.xlabel('Times')
plt.legend()
plt.show()