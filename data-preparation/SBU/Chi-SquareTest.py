from scipy import stats
import numpy as np
nframe = 50
depth = 16


def indices_augment(nframe, clip_len, ncandidate):
    indices_lst = []
    for i in range(ncandidate):
        indices = np.random.choice(nframe, clip_len, replace=False)
        indices.sort()
        p_value = stats.kstest(indices, stats.uniform(loc=0, scale=nframe).cdf)[1]
        indices_lst.append((indices, p_value))
    indices_lst.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
    indices = []
    print(len(indices_lst))
    for i in range(nframe // clip_len * 4):
        #print(-i-1)
        indices += indices_lst[i][0].tolist()
        print(indices_lst[i][1])
    return indices

lst = indices_augment(40, 16, 50)
lst.sort()
print(len(set(lst)))