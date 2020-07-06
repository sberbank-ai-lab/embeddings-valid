import os
import logging
from functools import reduce
from operator import iadd

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


RANDOM_SEED = 42
DATA_PATH = 'test_data'
N_TRAIN, N_VALID, N_TEST = 10000, 1000, 1000

N_FEATURES = [10, 20]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s')

    rs = np.random.RandomState(RANDOM_SEED)
    all_ids = np.arange(0, N_TRAIN + N_VALID + N_TEST) + 10000
    rs.shuffle(all_ids)

    id_train = all_ids[:N_TRAIN]
    id_valid = all_ids[N_TRAIN:N_TRAIN + N_VALID]
    id_test = all_ids[-N_TEST:]

    df_train = pd.DataFrame({'id': id_train}).sort_values('id')
    df_valid = pd.DataFrame({'id': id_valid}).sort_values('id')
    df_test = pd.DataFrame({'id': id_test}).sort_values('id')

    df_train.to_csv(os.path.join(DATA_PATH, 'id_train.csv'), index=False)
    df_valid.to_csv(os.path.join(DATA_PATH, 'id_valid.csv'), index=False)
    df_test.to_csv(os.path.join(DATA_PATH, 'id_test.csv'), index=False)

    df_target = []
    for i, n_features in enumerate(N_FEATURES):
        columns = [f'col_{j}' for j in range(n_features)]
        df_features = pd.DataFrame(rs.randn(all_ids.shape[0], n_features).round(3), columns=columns)
        col_id = 'id' if i != 0 else 'uid'
        df_features[col_id] = all_ids
        df_features.to_csv(os.path.join(DATA_PATH, f'features_{i}.csv'), index=False)

        for _ in range(i + 1):
            col_target = rs.choice(columns, 1)[0]
            df_target.append(df_features[col_target].clip(-1, None))

    df_target.append(pd.Series(np.random.randn(all_ids.shape[0])))
    df_target = reduce(iadd, df_target) / len(df_target)
    df_target = (df_target > 0.5).astype(int)

    df_target = pd.DataFrame({'id': all_ids, 'y': df_target}).sort_values('id')
    df_target.to_csv(os.path.join(DATA_PATH, 'target.csv'), index=False)
