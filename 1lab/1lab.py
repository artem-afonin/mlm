import csv
import os
from collections import namedtuple
from math import sqrt
from typing import List, Tuple

from sklearn.model_selection import train_test_split

# set path to resources and define default variables
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
res_dir = os.path.abspath(os.path.join(script_dir, "..", "res", "1lab"))

DataRow = namedtuple('DataRow', ['MrotInHour', 'Salary', 'Class'])
DataRow.__eq__ = lambda a, b: a.MrotInHour == b.MrotInHour and a.Salary == b.Salary and a.Class == b.Class


def read_csv(filepath: str) -> List[DataRow]:
    datarow_list = []
    with open(filepath, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            mrot, salary, clazz = int(row['MrotInHour']), int(row['Salary']), int(row['Class'])
            datarow_list.append(DataRow(mrot, salary, clazz))
    return datarow_list


# here is functions related to some evaluation and training
def leave_one_out(train_list: List[DataRow], max_h: int = 5) -> Tuple[int, List[Tuple]]:
    h_percent_list = []
    optimal_h = 0
    optimal_percent = 0

    for h in range(2, max_h + 1):
        correct_classification = 0
        wrong_classification = 0

        for main_row in train_list:
            class0_counter = 0
            class1_counter = 0
            for row in train_list:
                if not main_row == row:
                    if q_core(distance(main_row, row) / h) <= 1:
                        if row.Class == 0:
                            class0_counter += 1
                        else:
                            class1_counter += 1

            if class0_counter + class1_counter == 0:
                continue

            if class0_counter >= class1_counter:
                calculated_classification = 0
            else:
                calculated_classification = 1

            if main_row.Class == calculated_classification:
                correct_classification += 1
            else:
                wrong_classification += 1

            # print(f'\rH = {h} | correct = {correct_classification} | wrong = {wrong_classification}', end='',
            #       flush=True)

        percent = correct_classification / (correct_classification + wrong_classification)
        # print(f' -> {percent}')
        h_percent_list.append((h, percent))
        if optimal_percent < percent:
            optimal_percent, optimal_h = percent, h

    return optimal_h, h_percent_list


def q_core(r: float) -> float:
    return (1 - r ** 2) ** 2


def distance(a_row: DataRow, b_row: DataRow) -> float:
    return sqrt(
        ((a_row.MrotInHour - b_row.MrotInHour) ** 2)
        +
        ((a_row.Salary - b_row.Salary) ** 2)
    )


def parzen_window_with_fixed_h(train_list: List[DataRow], test_list: List[DataRow], h: int):
    class0_counter = 0
    class1_counter = 0
    correct_classification = 0

    for main_row in test_list:
        for row in train_list:
            if q_core(distance(main_row, row) / h) <= 1:
                if row.Class == 0:
                    class0_counter += 1
                else:
                    class1_counter += 1

        if (class0_counter >= class1_counter and main_row.Class == 0) or \
                (class1_counter > class0_counter and main_row.Class == 1):
            correct_classification += 1

    return correct_classification


def main():
    # My variant: 1
    # (1 + 2) % 5 + 1          == 4 data file
    # 1 % 3 + 1                == Parzen window method with fixed H
    # (1 * 6 + 13) % 8 % 3 + 1 == Q core function
    data_list = read_csv(os.path.join(res_dir, "data4.csv"))

    # split data onto two dataframes
    train_list, test_list = train_test_split(data_list, test_size=0.33)

    # find optimal h
    optimal_h, h_percent_list = leave_one_out(train_list, max_h=10)
    for h, percent in h_percent_list:
        print(f'H = {h} -> {percent}')
    print(f'\nOptimal H = {optimal_h}')

    # execute Parzen window method
    correct_classification = parzen_window_with_fixed_h(train_list, test_list, optimal_h)
    percent = correct_classification / len(test_list)
    print(f'\nFinal accuracy for Parzen window method with H = {optimal_h}: {percent}')


if __name__ == '__main__':
    main()
