import json

import numpy

from bm.apis.v1.APIsPredictionServices import NpEncoder
from bm.controllers.classification.ClassificationController import ClassificationController
from bm.db_helper.AttributesHelper import get_model_name, get_features, get_labels


class APIsClassificationServices:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def classify_data(self, content):
        testing_values = []
        features_list = get_features()
        for i in features_list:
            feature_value = str(content[i])
            final_feature_value = feature_value  # float(feature_value) if feature_value.isnumeric() else feature_value
            testing_values.append(final_feature_value)
        classification_controller = ClassificationController()
        text_category = [numpy.array(classification_controller.classify_text(feature_value))]

        # Create predicted values json object
        lables_list = get_labels()
        text_category_json = {}
        for j in range(len(text_category)):
            for i in range(len(lables_list)):
                bb = text_category[j][i]
                text_category_json[lables_list[i]] = text_category[j][i]
                # NpEncoder = NpEncoder(json.JSONEncoder)
            json_data = json.dumps(text_category_json, cls=NpEncoder)

        return json_data

    def classify_data_list(self, content):
        text_category_json = {}
        for i in range(len(content)):
            class_item = self.classify_data(content[i])
            text_category_json[i] = class_item
        json_data = json.dumps(text_category_json, cls=NpEncoder)

        return json_data

    def get_reports(self, content):
        classification_controller = ClassificationController()
        report_name, params = classification_controller.get_reports_list(content['data'])
        report_name = numpy.array(report_name)
        report_name = report_name.flatten()
        # {
        #         "report_name": "xyz",
        #         "report_params": {
        #             "param_1": "1",
        #             "param_2": "1",
        #             "param_3": "1"
        #         }
        # }

        # Create predicted values json object
        report_profile_json = {}
        report_profile_json["report_name"] = report_name[0]
        text_params_json= {}
        for key, value  in params.items():
            text_params_json[key] = value
        report_profile_json["report_params"] = text_params_json
        json_data = json.dumps(report_profile_json, cls=NpEncoder)

        return json_data