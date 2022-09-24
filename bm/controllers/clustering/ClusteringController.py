import os
import os
import pickle
import random
from datetime import datetime
from random import randint

import numpy as np
import pandas as pd
# from app import config_parser
from flask import session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

from app import db, config_parser
from app.base.constants.BM_CONSTANTS import app_root_path
from app.base.db_models.ModelProfile import ModelProfile
from base.constants.BM_CONSTANTS import pkls_location, plot_locations, plot_zip_locations, \
    clustering_root_path, data_files_folder, df_location
from bm.controllers.ControllersHelper import ControllersHelper
from bm.controllers.clustering.ClusteringControllerHelper import ClusteringControllerHelper
from bm.core.ModelProcessor import ModelProcessor
from bm.db_helper.AttributesHelper import add_features, add_labels, add_api_details, \
    update_api_details_id
from bm.utiles.Helper import Helper


class ClusteringController:
    members = []

    file_name = config_parser.get('SystemConfigurations', 'SystemConfigurations.default_data_file_prefix')

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def get_data_cluster(self, data):
        """
        Get the correct cluster of provided data and related keywords
        @param data:
        @return: clusters, keywords
        """
        try:
            # load the vectorizer
            vic_filename = '%s%s%s' % (app_root_path, pkls_location, 'vectorized_pkl.pkl')
            vectorizer = pickle.load(open(vic_filename, 'rb'))

            # Load the model
            model_file_name = pkls_location + self.file_name + '_model.pkl'
            cls = pickle.load(open(model_file_name, 'rb'))

            # Get custer/s
            clusters = cls.predict(vectorizer.transform(data))

            # Get clusters keywords
            keywords = []
            clusters_dic = {}
            for cluster in clusters:
                keywords.append(ClusteringControllerHelper.get_cluster_keywords(cluster))

            for i in range(len(clusters)):
                keywords_str = keywords[i]
                cluster_key = 'Cluster_' + str(clusters[i])
                clusters_dic[cluster_key] = keywords_str

            return clusters_dic
        except  Exception as e:
            return 0

    def run_clustering_model(self, location_details, ds_goal, ds_source, is_local_data, featuresdvalues=['data']):
        try:
            # ------------------Preparing data frame-------------------------#
            model_id = randint(0, 10)
            helper = Helper()

            # Prepare the date and creating the clustering model
            clusteringcontrollerhelper = ClusteringControllerHelper()
            files_path = '%s%s' % (clustering_root_path, data_files_folder)
            # Create datafile (data.pkl)
            if (is_local_data == 'Yes'):
                folders_list = ControllersHelper.get_folder_structure(files_path, req_extensions=('.txt'))
                featuresdvalues = ['data']
                data_set = clusteringcontrollerhelper.create_clustering_data_set(files_path)
            elif (is_local_data == 'csv'):
                csv_file_path = '%s%s' % (df_location, session['fname'])
                data_set = clusteringcontrollerhelper.create_clustering_csv_data_set(csv_file_path, featuresdvalues)
            else:
                folders_list = helper.list_ftp_dirs(
                    location_details)
                data_set = clusteringcontrollerhelper.create_clustering_FTP_data_set(location_details)

            full_file_path = '%s%s' % (df_location, 'data.pkl')

            X_train = pd.read_pickle(full_file_path)
            X_train = shuffle(X_train)
            # dcp = DataCoderProcessor()
            # real_x = dcp.vectrise_feature_text(model_id, X_train)
            # pca = PCA(2)
            # Transform the data
            # real_x = pca.fit_transform(real_x)
            documents = X_train[featuresdvalues].values.astype("U")
            documents = documents.flatten()
            features = []
            vectorizer = TfidfVectorizer(stop_words='english')

            # for feature_array in X_train.columns:
            #     if X_train[feature_array].dtypes == np.object:

            features = vectorizer.fit_transform(documents)

            # Store vectorized
            vic_filename = '%s%s%s' % (app_root_path, pkls_location, 'vectorized_pkl.pkl')
            pickle.dump(vectorizer, open(vic_filename, 'wb'))

            # Select proper model
            mp = ModelProcessor()
            cls = mp.clustering_model_selector()
            model = cls.fit(features)
            y_pred = cls.predict(features)
            X_train['cluster'] = model.labels_
            data_file_location = ClusteringControllerHelper.generate_labeled_datafile(session['fname'],
                                                                                      model.labels_)  # Add label column to orginal data file

            model_file_name = pkls_location + self.file_name + '_model.pkl'
            pickle.dump(cls, open(model_file_name, 'wb'))

            # Delete old visualization images
            dir = plot_locations
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
            for f in os.listdir(plot_zip_locations):
                os.remove(os.path.join(plot_zip_locations, f))

            # Show Elbow graph and get clusters' keywords
            html_path = ClusteringControllerHelper.plot_elbow_graph(features.data)
            # html_path = ClusteringControllerHelper.plot_clustering_report(features.data, model, model.labels_, file_name)
            clusters_keywords = ClusteringControllerHelper.extract_clusters_keywords(model, 5, vectorizer)

            # ------------------Predict values from the model-------------------------#
            now = datetime.now()
            all_return_values = {'file_name': self.file_name,
                                 'clusters_keywords': clusters_keywords,
                                 'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'data_file_location': data_file_location}

            # Add model profile to the database
            modelmodel = {'model_id': model_id,
                          'model_name': self.file_name,
                          'user_id': 1,
                          'model_headers': 'str(cvs_header)[1:-1]',
                          'prediction_results_accuracy': 'str(c_m)',
                          'mean_absolute_error': 'str(Mean_Absolute_Error)',
                          'mean_squared_error': 'str(Mean_Squared_Error)',
                          'root_mean_squared_error': 'str(Root_Mean_Squared_Error)',
                          'plot_image_path': html_path,
                          'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'ds_source': ds_source,
                          'ds_goal': ds_goal}
            model_model = ModelProfile(**modelmodel)
            # Delete current profile
            model_model.query.filter().delete()
            db.session.commit()
            # Add new profile
            db.session.add(model_model)
            db.session.commit()

            # Add features, labels, and APIs details
            add_features_list = add_features(model_id, [self.file_name])
            add_labels_list = add_labels(model_id, ['cluster'])
            api_details_id = random.randint(0, 22)
            api_details_list = add_api_details(model_id, api_details_id, 'v1')
            api_details_list = update_api_details_id(api_details_id)

            # APIs details and create APIs document

            return all_return_values
        except Exception as e:
            return {}
            # return config_parser.get('ErrorMessages', 'ErrorMessages.fail_create_model')
