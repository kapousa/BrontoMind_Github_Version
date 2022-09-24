import logging
import os

import numpy
from flask import session
from mailmerge import MailMerge

from app import db
from app.base.db_models.ModelProfile import ModelProfile
from bm.apis.v1.APIsPredictionServices import predictvalues, getmodelfeatures, getmodellabels, nomodelfound
from bm.db_helper.AttributesHelper import get_model_name, get_features, get_labels

from app.base.db_models.ModelAPIMethods import ModelAPIMethods
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from docxcompose.composer import Composer
from docx import Document as Document_compose


class APIHelper:

    def api_runner(self, content):
        # test git

        model_name = get_model_name()
        if get_model_name() != None:
            serv = content['serv']
            apireturn_json = {}
            if serv == 'predictevalues':
                inputs = content['inputs']
                apireturn_json = predictvalues(inputs)
            elif serv == 'getmodelfeatures':
                apireturn_json = getmodelfeatures()
            elif serv == 'getmodellabels':
                apireturn_json = getmodellabels()
            else:
                aa = 0
            return apireturn_json
        else:
            return nomodelfound()

    def generate_api_details(self):
        api_details = []
        return api_details

    def generateapisdocs(self, model_name, base_url, templates_folder, output_pdf_folder):
        """
        Generate the API document according to the model type 
        :param model_name: 
        :param base_url: 
        :param templates_folder: 
        :param output_pdf_folder: 
        :return: Success = 1, Fail= error
        """
        aa = session['ds_goal']
        api_methods_ids = ModelAPIMethods.query.with_entities(ModelAPIMethods.api_method_id).filter(ModelAPIMethods.model_goal == session['ds_goal']).all()
        api_methods_ids_arr = numpy.array(api_methods_ids).flatten()
        generate_apis_request_sample = self.generate_api_method_reqres_samples(api_methods_ids_arr)

        try:
            apis_doc_cover_template = templates_folder + "Slonos_Labs_BrontoMind_APIs_document_cover_template.docx"
            output_cover_file = str(output_pdf_folder + model_name + '_BrontoMind_APIs_cover_document.docx')

            apis_doc_template = templates_folder + "Slonos_Labs_BrontoMind_APIs_document_template.docx"
            output_methods_file = str(output_pdf_folder + model_name + '_BrontoMind_APIs_methods_document.docx')

            output_file = str(output_pdf_folder + model_name + '_BrontoMind_APIs_document.docx')
            output_pdf_file = str(output_pdf_folder + model_name + '_BrontoMind_APIs_document.pdf')

            # 1- Adding the cover
            output_cover_contents = []
            api_details = ModelAPIDetails.query.first()
            cover_pages = MailMerge(apis_doc_cover_template)
            cover_pages.merge(version=api_details.api_version,
                api_version=api_details.api_version,
                public_key=api_details.public_key,
                private_key=api_details.private_key)
            cover_pages.write(output_cover_file)

            # 2-Adding the methods
            output_contents = []
            for api_method_id in api_methods_ids_arr:
                api_method_id = str(api_method_id)
                api_method = ModelAPIMethods.query.filter(ModelAPIMethods.api_method_id == api_method_id).first()
                api_methods_dict = {
                    "method_name": api_method.method_name,
                    "method_description": api_method.method_description,
                    "url": str(base_url + api_method.url),
                    "sample_request": api_method.sample_request,
                    "sample_response": api_method.sample_response
                }

                output_contents.append(api_methods_dict)

            document_merge = MailMerge(apis_doc_template)
            document_merge.merge_templates(output_contents, separator='textWrapping_break')
            document_merge.write(output_methods_file)

            # 3- Merge cover with methods document
            self.create_api_document(output_cover_file, output_methods_file, output_file)

            # 4- convert the file to PDF format
            #docxtopdf = DocxToPDF()
            #print("Processing...")
            #docxtopdf.convert(output_file, output_pdf_file)
            #print("Processed...")
            #os.remove(output_file)

            return 1

        except Exception as e:
            logging.error("generateapisdocs\n" + e)

    def create_api_document(self, filename_master, filename_second, final_filename):
        """
        Create the API document
        :param filename_master:
        :param filename_second:
        :param final_filename:
        :return:
        """
        # filename_master is name of the file you want to merge the docx file into
        master = Document_compose(filename_master)

        composer = Composer(master)
        # filename_second_docx is the name of the second docx file
        doc2 = Document_compose(filename_second)
        # append the doc2 into the master using composer.append function
        composer.append(doc2)
        # Save the combined docx with a name
        composer.save(final_filename)

        # Delete sub files
        os.remove(filename_master)
        os.remove(filename_second)


        return 1

    def generate_api_method_reqres_sample(self, api_method_id=1):
        """
        Generate the request/response's sample of the generated API
        :param api_method_id
        :return 1: Success, 0: Fail:
        """
        try:
            # Generate the sample request
            modelfeatures = get_features()
            sample_request = "{\n"

            for i in range(len(modelfeatures)):
                sample_request+= "%s%s%s:" % ('"',modelfeatures[i],'"')
                if i < len(modelfeatures) - 1:
                    sample_request+= "'',\n"
                else:
                    sample_request += "\n"
            sample_request+= "}"

            # Update the method's sample request
            api_method_id = str(api_method_id)
            modelapimethods = ModelAPIMethods.query.filter_by(api_method_id = api_method_id).first()
            modelapimethods.sample_request = sample_request

            # Generate the sample response
            modellabels = get_labels()
            sample_response = "{\n"

            for i in range(len(modellabels)):
                sample_response += "%s%s%s:" % ('"', modellabels[i], '"')
                if i < len(modellabels) - 1:
                    sample_response += "'',\n"
                else:
                    sample_response += "\n"
            sample_response += "}"
            # Update the method's sample request
            modelapimethods.sample_response = sample_response

            db.session.commit()

            return 1
        except Exception as e:
            logging.error("generate_apis_request_sample\n" + e)

    def generate_api_method_reqres_samples(self, api_methods_ids=[1]):
        """
        Generate the request/response's samples of the generated API
        :param api_methods_ids:
        :return: 1: Success, 0: Fail:
        """
        try:
            for i in range(len(api_methods_ids)):
                api_method_reqres_sample = self.generate_api_method_reqres_sample(api_methods_ids[i])
            return 1
        except Exception as e:
            logging.error("generate_api_method_reqres_samples\n" + e)
