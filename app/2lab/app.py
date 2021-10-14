import os
from argparse import ArgumentParser

import graphviz
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
res_dir = os.path.abspath(os.path.join(script_dir, '../..', 'res', '2lab'))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    show_graphs = args.graph

    df = pd.read_csv(os.path.join(res_dir, 'heart_data.csv'))
    df = df.replace('?', None)

    table = PrettyTable()
    table.field_names = ['iterations', 'depth', 'percent']
    d = {}

    for iteration in range(10):
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        train_result_column = train_df['goal']
        train_df = train_df.drop(columns=['goal'])
        test_result_column = test_df['goal']
        test_df = test_df.drop(columns=['goal'])

        for depth in range(2, 16 + 1):
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(train_df, train_result_column)

            test_result = clf.predict(test_df)

            score = accuracy_score(test_result_column, test_result)
            try:
                d[iteration]
            except KeyError:
                d[iteration] = []
            d[iteration].append((depth, score))

            if show_graphs and iteration == 0:
                r = export_graphviz(clf, feature_names=train_df.columns, class_names=['healthy', 'cancer'])
                graph = graphviz.Source(r)
                png_bytes = graph.pipe(format='png')
                with open(f'graphviz_{depth}.png', 'wb') as f:
                    f.write(png_bytes)

    for iteration, pair_list in d.items():
        depth, max_score = max(*pair_list, key=lambda pair: pair[1])
        table.add_row([iteration + 1, depth, '%.2f%%' % (100 * max_score)])

    print(table)


if __name__ == '__main__':
    main()
