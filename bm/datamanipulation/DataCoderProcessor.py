#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import csv

import numpy
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from app import db
from sklearn import preprocessing

from app.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from bm.db_helper.AttributesHelper import get_encoded_columns
from bm.utiles.CVSReader import getcvsheader


class DataCoderProcessor:
    pkls_location = 'pkls/'
    category_location = 'app/data/'
    flag = ''

    def __init__(self):
        self.flag = '_'

    def encode_features(self, model_id, data: DataFrame, column_type='F'):
        columns_name = data.columns
        encoded_columns = []
        data_types = data.dtypes
        for i in range(len(data_types)):
            if data_types[i] != np.int64 and data_types[i] != np.float:
                data_item = {'model_id': model_id, 'column_name': columns_name[i],
                             'column_type': column_type}
                encoded_columns.append(data_item)
                col_name = columns_name[i]
                dummies = self.encode_column(col_name, data[[col_name]])
                dummies = pd.DataFrame(dummies)
                data = data.drop([col_name], axis=1)
                data.insert(i, col_name, dummies)

        db.session.bulk_insert_mappings(ModelEncodedColumns, encoded_columns)
        db.session.commit()
        db.session.close()

        return data

    def encode_labels(self, model_id, data: DataFrame):
        column_type = 'L'
        return self.encode_features(model_id, data, 'L')

    def vectrise_feature_text(self, model_id, data: DataFrame):
        columns_name = data.columns
        encoded_columns = []
        data_types = data.dtypes
        for i in range(len(data_types)):
            if data_types[i] != np.int64 and data_types[i] != np.float:
                data_item = {'model_id': model_id, 'column_name': columns_name[i],
                             'column_type': 'F'}
                encoded_columns.append(data_item)
                col_name = columns_name[i]
                dummies = self.vectorize_column(col_name, data[[col_name]])
                dummies = pd.DataFrame(dummies)
                data = data.drop([col_name], axis=1)
                data.insert(i, col_name, dummies[1:])

        db.session.bulk_insert_mappings(ModelEncodedColumns, encoded_columns)
        db.session.commit()
        db.session.close()

        return data

    def encode_input_values(self, features_list, input_values):
        encoded_columns = []
        model_encoded_columns = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(
                ModelEncodedColumns.column_type == 'F').all())
        model_encoded_columns = model_encoded_columns.flatten()
        for i in range(len(input_values)):
            input_value = input_values[i].strip()
            if (not input_value.isdigit()) and (features_list[i] in model_encoded_columns):
                col_name = features_list[i]
                pkl_file_location = self.pkls_location + col_name + '_pkle.pkl'
                encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
                column_data_arr = numpy.array(input_value)
                encoded_values = encoder_pkl.transform(column_data_arr.reshape(-1, 1))
                encoded_columns.append(encoded_values[0])
            else:
                encoded_columns.append(input_value)

        return [encoded_columns]

    def decode_output_values(self, labels_list, input_values):
        decoded_results = []
        decoded_row = []
        model_encoded_columns = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(
                ModelEncodedColumns.column_type == 'L').all())
        model_encoded_columns = model_encoded_columns.flatten()
        for i in range(len(input_values)):
            input_values_row = input_values[i]
            for j in range(len(input_values[i])):
                if labels_list[j] in model_encoded_columns:
                    col_name = labels_list[j]
                    pkl_file_location = self.pkls_location + col_name + '_pkle.pkl'
                    encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
                    column_data_arr = numpy.array(input_values_row[j])
                    original_value = encoder_pkl.inverse_transform(column_data_arr.reshape(-1, 1))
                    decoded_row.append(original_value[0].strip())
                else:
                    decoded_row.append(str(input_values_row[j]))
            decoded_results.append(decoded_row)
        return np.array(decoded_results)

    def decode_category_name(self, category_column, input_values):
        pkl_file_location = self.pkls_location + category_column + '_pkle.pkl'
        encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
        original_value = encoder_pkl.inverse_transform(input_values)
        return original_value

    def encode_column(self, column_name, column_data):
        column_data_arr = column_data.to_numpy()
        column_data_arr = column_data_arr.flatten()
        categories = numpy.unique(column_data_arr)
        labelEnc = preprocessing.LabelEncoder()
        encoded_values = labelEnc.fit_transform(column_data_arr.reshape(-1, 1))
        pkl_file_location = self.pkls_location + column_name + '_pkle.pkl'
        pickle.dump(labelEnc, open(pkl_file_location, 'wb'))

        # save categories
        category_file_location = self.category_location + column_name + '_csv.csv'
        with open(category_file_location, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(categories)

        return encoded_values

    def vectorize_column(self, column_name, column_data):
        column_data_arr = column_data.to_numpy()
        column_data_arr = column_data_arr.flatten()
        categories = numpy.unique(column_data_arr)
        vectorizer = TfidfVectorizer()
        column_data_arr = column_data_arr.flatten()
        column_data_list = list(column_data_arr)
        vectors = vectorizer.fit_transform(column_data_list)
        pkl_file_location = self.pkls_location + column_name + '_pkle.pkl'
        pickle.dump(vectorizer, open(pkl_file_location, 'wb'))

        # save categories
        category_file_location = self.category_location + column_name + '_csv.csv'
        with open(category_file_location, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(categories)

        return vectors.indptr

    def get_all_gategories_values(self=None):
        encoded_columns = get_encoded_columns('F')

        if len(encoded_columns) == 0:
            return '0'

        all_gategories_values = {}
        for i in range(len(encoded_columns)):
            category_file_location = DataCoderProcessor.category_location + encoded_columns[i] + '_csv.csv'
            category_values = getcvsheader(category_file_location)
            all_gategories_values[encoded_columns[i]] = category_values
        return all_gategories_values
