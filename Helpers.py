import csv
import numpy as np


def read_from_csv_cat(path):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        dataset = []
        header = next(csv_reader)
        for row in csv_reader:
            values = []
            for value in row:
                values.append(value)
            dataset.append(values)
        return header, np.array(dataset, dtype=object)


def read_from_csv_num(path):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        dataset = []
        header = next(csv_reader)
        for row in csv_reader:
            values = []
            for i in range(len(row)-1):
                values.append(float(row[i]))
            values.append(row[len(row)-1])
            dataset.append(values)
        return header, np.array(dataset, dtype=object)


def read_from_csv_mixed(path):
    """For the datasets with both continuous and categorical values, the position of the numerical ones must be known.
    In the case of heart_disease_cleveland the positions of the attributes to be transformed in float are 0,3,4,7,9"""
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        dataset = []
        header = next(csv_reader)
        for row in csv_reader:
            values = []
            for i in range(len(row)):
                if i in [0,3,4,7,9]:
                    values.append(float(row[i]))
                else:
                    values.append(row[i])
            dataset.append(values)
        return header, np.array(dataset, dtype=object)

