# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_login import login_required

from app.base.app_routes.directors.ClusteringDirector import ClusteringDirector
from app.clustering import blueprint

## Clustering

@blueprint.route('/downloadlabledfile', methods=['GET', 'POST'])
@login_required
def downloadlabledfile():
    return ClusteringDirector.download_labeled_datafile()
