import logging
import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
res_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'res', '3lab'))

log = logging.info


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def read_dataframe_from_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns='type')
    df = df.astype(np.float32)
    df = df.fillna(method='ffill')
    df = (df - df.min()) / (df.max() - df.min())
    return df


def split_wine_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return df, df[df['type'] == 'white'], df[df['type'] == 'red']


def fit_and_predict(df: pd.DataFrame, iterations=10) -> List[float]:
    results = []

    for i in range(iterations):
        log(f'iteration {i + 1}')
        train_df, test_df = train_test_split(df, test_size=0.3)

        train_predict = train_df['quality']
        train_df = train_df.drop(columns=['quality'])
        test_predict = test_df['quality']
        test_df = test_df.drop(columns=['quality'])

        clf = LassoCV()
        clf.fit(train_df, train_predict)
        test_result = clf.predict(test_df)

        acc = 0
        for i in range(len(test_predict)):
            acc += abs(test_result[i] - test_predict.iloc[i])
        bad = acc / len(test_predict)
        results.append(1 - bad)

    return results


def print_results_table(result_dict: Dict[str, List]):
    table = PrettyTable()
    table.field_names = ('wine type', 'iterations', 'percent')
    for type_name, type_results in result_dict.items():
        for iteration, score in enumerate(type_results, 1):
            table.add_row((type_name, iteration, '%.2f%%' % (100 * score)))
        table.add_row(('', '', ''))

    # dirty hack for deleting last blank row (:
    table.del_row(-1)
    print(table)


def main():
    setup_logging()

    df = read_dataframe_from_csv(os.path.join(res_dir, 'wine_data.csv'))
    log('read csv data successfully')

    all_df, white_df, red_df = split_wine_df(df)
    all_df = process_dataframe(all_df)
    white_df = process_dataframe(white_df)
    red_df = process_dataframe(red_df)
    log('data processed and prepared for fit and predict')

    log('processing "all" dataframe...')
    all_df_results = fit_and_predict(all_df)

    log('processing "white wine" dataframe...')
    white_df_results = fit_and_predict(white_df)

    log('processing "red wine" dataframe...')
    red_df_results = fit_and_predict(red_df)

    log('end fit and predict stage')
    log('printing all results table')
    print_results_table({
        'all': all_df_results,
        'white': white_df_results,
        'red': red_df_results
    })

    # TODO: calculate metrics


if __name__ == '__main__':
    main()
