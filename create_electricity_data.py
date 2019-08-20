import csv
import numpy as np


# path = '/home/long/Desktop/DDL/projects/self-boosted-timeseries/datasets/LD2011-2014/test.csv'
path = '/home/long/Desktop/DDL/projects/self-boosted-timeseries/datasets/LD2011-2014/LD2011_2014.txt'


remove_comma = lambda t: t.replace(',', '.')


with open(path) as file_pointer:
    reader = csv.reader(file_pointer, delimiter=';')

    with open('data/clean_electricity.csv', 'w', newline='') as file_writer:
        writer = csv.writer(file_writer)
        for idx, row in enumerate(reader):
            if idx < 1:
                writer.writerow(['time, avg_electricity'])
                continue

            my_date = row[0]
            removed_commas = np.array([remove_comma(xi) for xi in row[1:]])

            numerical_data = np.asarray(removed_commas, dtype=np.float32)
            non_zero = numerical_data[numerical_data != 0]
            if len(non_zero) < 1:
                print('dropping row:', idx)
                continue
            # avg = np.sum(numerical_data, axis=0) / len(numerical_data)
            avg2 = np.mean(non_zero)
            writer.writerow([my_date, avg2])
            print("datetime:", my_date, "average:", avg2)
