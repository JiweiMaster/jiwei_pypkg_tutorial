import numpy as np

# 将数据集合分成k折
def kfold_list(n, K, seed=123):
    rnd_state = np.random.RandomState(seed)
    dataset_rnd_sort = np.array(list(range(n)))
    rnd_state.shuffle(dataset_rnd_sort)
    idxlist = list(range(0, n, 1))
    val_idx = [list(range(i*(n//K), (i+1)*(n//K))) for i in range(0, K, 1)]
    train_idx = [list(range(0, i*int(n//K), 1))+list(range((i+1)*(n//K), n, 1)) for i in range(0, K, 1)]
    if val_idx[-1][-1] < n-1:
        val_idx[-1] = val_idx[-1]+list(range(val_idx[-1][-1]+1, n, 1))
        train_idx[-1] = list(range(0, (K-1)*(n//K)))
    val_from_list = []
    train_from_list = []
    for ival, itrain in zip(val_idx, train_idx):
        val_from_list.append(dataset_rnd_sort[ival])
        train_from_list.append(dataset_rnd_sort[itrain])
    return train_from_list, val_from_list