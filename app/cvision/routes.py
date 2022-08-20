# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template
from app.cvision import blueprint

## CVision

@blueprint.route('/')
def index():
    return 0
