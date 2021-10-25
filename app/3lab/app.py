import os

import numpy as np
import pandas as pd

from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
res_dir = os.path.abspath(os.path.join(script_dir, '../..', 'res', '3lab'))


def main():
    df = pd.read_csv(os.path.join(res_dir, 'wine_data.csv'))
    df.fillna(0, inplace=True)
    df = df.replace('white', 0)
    df = df.replace('red', 1)

    table = PrettyTable()
    table.field_names = ['iterations', 'percent']
    d = {}

    for iteration in range(10):
        train_df, test_df = train_test_split(df, test_size=0.3)

        train_result_column = train_df['quality']
        train_df = train_df.drop(columns=['quality'])
        test_result_column = test_df['quality']
        test_df = test_df.drop(columns=['quality'])

        train_df = train_df.astype(float)
        test_df = test_df.astype(float)

        train_df = preprocessing.normalize(train_df)
        test_df = preprocessing.normalize(test_df)

        clf = LassoCV()
        clf.fit(train_df, train_result_column)
        test_result = clf.predict(test_df)

        test_result = np.around(test_result)
        # print(test_result)

        score = accuracy_score(test_result_column, test_result)
        try:
            d[iteration]
        except KeyError:
            d[iteration] = []
        d[iteration].append(score)

    for iteration, score in d.items():
        max_score = max(score)
        table.add_row([iteration + 1, '%.2f%%' % (100 * max_score)])

    print(table)


if __name__ == '__main__':
    main()
