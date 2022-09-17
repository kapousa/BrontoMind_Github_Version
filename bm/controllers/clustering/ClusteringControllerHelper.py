import os
import pickle
from random import shuffle

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly as pl
import plotly.graph_objs as pgo
from nltk.corpus import stopwords
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from app.base.constants.BM_CONSTANTS import html_plots_location, html_short_path, pkls_location, \
    clustering_root_path, df_location
from bm.controllers.ControllersHelper import ControllersHelper
from bm.core.engine.processors.ClusteringModelProcessor import ClusteringModelProcessor
from bm.utiles.Helper import Helper


class ClusteringControllerHelper:
    # stop_words = set(stopwords.words('english'))

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.stop_words = set(stopwords.words('english'))

    def plot_clustering_report_(df, model, label, model_name='file_name'):
        # Represent neighborhoods as in previous bubble chart, adding cluster information under color.
        df = pd.DataFrame(df)
        df['label'] = label

        x_axis = np.array(df[0])
        y_axis = np.array(df[1])
        x_center = numpy.array(model.cluster_centers_[:, 0])
        y_center = numpy.array(model.cluster_centers_[:, 1])
        u_label = np.unique(label)

        fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.scatter(x_axis, y_axis, s=50)
        plt.scatter(x_center, y_center, marker='x')
        fig.show()
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        pl.offline.plot(fig, filename=html_file_location, config={'displayModeBar': True}, auto_open=False)

        return html_path

    @staticmethod
    def plot_clustering_report(df, model, label, model_name='file_name'):
        # Represent neighborhoods as in previous bubble chart, adding cluster information under color.
        df = pd.DataFrame(df)
        df['label'] = label
        u_label = np.unique(label)
        trace_arr = []
        for i in u_label:
            label_data = df.loc[df['label'] == i]
            trace0 = pl.graph_objs.Scatter(x=label_data[0],
                                 y=label_data[1],
                                 text="Cluster:" + str(i),
                                 marker=pgo.scatter.Marker(symbol='x',
                                                   size=12,
                                                   color=u_label, opacity=0.5),
                                 showlegend=False
                                 )
            trace_arr.append(trace0)

        # Represent cluster centers.
        trace1 = pl.graph_objs.Scatter(x=model.cluster_centers_[:, 0],
                                       y=model.cluster_centers_[:, 1],
                                       name='',
                                       mode='markers',
                                       marker=pgo.scatter.Marker(symbol='x',
                                                                 size=12,
                                                                 color='black'),
                                       showlegend=False
                                       )
        trace_arr.append(trace1)

        layout5 = pgo.Layout(title='Baltimore Vital Signs (PCA)',
                             xaxis=pgo.layout.XAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             yaxis=pgo.layout.YAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             hovermode='closest')

        data7 = pgo.Data(trace_arr)
        layout7 = layout5
        layout7['title'] = 'Baltimore Vital Signs (PCA and k-means clustering with 7 clusters)'
        fig7 = pgo.Figure(data=data7, layout=layout7)

        # Plot model
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        fig = make_subplots(rows=2, cols=1)
        # 1- clusters
        for k in range(len(fig7.data)):
            fig.add_trace(fig7.data[k], row=1, col=1)
        # 2- Elbow graph
        # TODO: Add code to implent Elbow function and plot it with the clustering graph

        pl.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    def train_clusterfer(self, docs, categories):
        try:
            X_train = self.get_clusterfer_splits(docs)

            vectorized = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')

            # Create doc-term matrix
            dtm = vectorized.fit_transform(X_train)

            # Train the model
            clusteringmodelselector = ClusteringModelProcessor()
            cls = clusteringmodelselector.clusteringmodelselector(6)

            # Model evaluations
            controller_helper = ControllersHelper()
            train_precision, train_recall, train_f1 = self.evaluate_clustfer(cls, vectorized,
                                                                             X_train, y_train, categories)
            test_precision, test_recall, test_f1 = self.evaluate_clustfer(cls, vectorized, X_test,
                                                                          y_test, categories)

            # store the classifier
            clf_filename = '%s%s%s' % (clustering_root_path, pkls_location, 'cluterfer_pkl.pkl')
            pickle.dump(cls, open(clf_filename, 'wb'))

            # Store vectorized
            vic_filename = '%s%s%s' % (clustering_root_path, pkls_location, 'vectorized_pkl.pkl')
            pickle.dump(vectorized, open(vic_filename, 'wb'))

            precision = numpy.array([train_precision, test_precision])
            recall = numpy.array([train_recall, test_recall])
            f1 = numpy.array([train_f1, test_f1])

            all_return_value = {'train_precision': str(precision),
                                'train_recall': str(recall),
                                'train_f1': str(f1),
                                'test_precision': str(precision),
                                'test_recall': str(recall),
                                'test_f1': str(f1)}

            return all_return_value

        except  Exception as e:
            print(e)
            return e

    def evaluate_clustfer(self, classifier, vectorizer, X_test, y_test, categories):
        return 0, 1, 2

    def get_clusterfer_splits(self, docs):
        shuffle(docs)
        return docs

    def create_clustering_data_set(self, files_path):
        try:
            output_file = '%s%s' % (files_path, 'data.pkl')
            files_data = {}
            if os.path.exists(output_file):
                os.remove(output_file)

            with open(output_file, 'w', encoding='utf8') as outfile:
                for filename in os.listdir(files_path):
                    with open(filename, 'rb') as file:
                        text = file.read().decode(errors='replace').replace('\n', '')
                        files_data.append(text)
                        outfile.write('%s\n' % (text))
            df = pd.DataFrame(files_data)
            df.to_pickle(outfile)

            return 1
        except  Exception as e:
            print(e)
            return 0

    def create_clustering_csv_data_set(self, csv_file_path, features_list):
        try:
            df = pd.read_csv(csv_file_path)
            df = df.loc[:, features_list]
            output_file = '%s%s' % (df_location, 'data.pkl')
            # file_data = []
            # if os.path.exists(output_file):
            #     os.remove(output_file)
            # # Create reader object by passing the file
            # # object to reader method
            # with open(csv_file_path, 'r') as read_obj:
            #     csv_dict_reader = DictReader(read_obj)
            #     for row in csv_dict_reader:
            #         data_row = []
            #         for key, value in row.items():
            #             data_row.append(value)
            #         if (data_row[0] != ''):
            #             file_data.append(value)
            # df = pd.DataFrame(file_data)
            df.to_pickle(output_file)

            return 1
        except  Exception as e:
            print(e)
            return 0

    def create_clustering_FTP_data_set(self, location_details):
        try:
            output_file = '%s%s' % (df_location, 'data.pkl')
            helper = Helper()
            ftp_conn = helper.create_FTP_conn(location_details)
            files_data = []

            # Reading files contents and save it in pkl file
            ftp_conn.cwd(df_location)
            files_list = ftp_conn.nlst()
            for filename in files_list:
                fullfilename = filename
                gFile = open("temp.txt", "wb")
                ftp_conn.retrbinary(f"RETR {fullfilename}", gFile.write)
                gFile.close()
                with open("temp.txt", 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', '')
                    files_data.append(text)
                gFile.close()
            ftp_conn.quit()

            df = pd.DataFrame(files_data)
            df.to_pickle(output_file)

            return 1
        except  Exception as e:
            print(e)
            return 0
