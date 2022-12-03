# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelQaxessReports(db.Model):

    __tablename__ = 'qaxess_reports'

    id = Column(Integer, primary_key=True, unique=True)
    report_name = Column(String)
    description = Column(String)
    parameters = Column(String)
    rep_par = Column(String)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)



