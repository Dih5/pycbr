#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
import logging.config
import os
import json

import coloredlogs
import yaml
import pandas as pd

from flask import Flask, request
from flask_cors import CORS
from flask_restplus import Api, Resource, fields

from . import __version__


def setup_logging(default_path='logging.yaml', env_key='CBR_LOG', default_level=logging.INFO):
    """
    Args:
        default_path (str): A path to a yaml file with the logging configuration.
        env_key (str): The name of an environment variable with a path to the logging file.
                       Has preference over default_path.
        default_level (int): A level of logging (e.g., logging.INFO) used in case an error occurs.

    Returns:
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in logging configuration file. Using the default settings.')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Unable to load a logging configuration file. Using the default settings.')


logger = logging.getLogger(__name__)
setup_logging()


class CBR_Flask(Flask):
    def __init__(self, import_name, case_base, recovery, aggregator):
        super().__init__(import_name)
        CORS(self)

        self.api = Api(app=self, version=__version__, title="pyCBR", description="pyCBR generated API REST")
        self.api_namespace = self.api.namespace("api", description="General methods")

        self.config.update(
            ERROR_404_HELP=False,  # No "but did you mean" messages
            RESTPLUS_MASK_SWAGGER=False,
        )

        # Log POST bodies
        @self.before_request
        def log_request_info():
            # self.logger.debug('Headers: %s', request.headers)
            data = request.get_data()
            if data:
                self.logger.info('Body: %s', data.decode())

        self.case_base = case_base

        self.models = {}
        models = self.models

        self.models["version"] = self.api.model('Deployment configuration', {
            'version': fields.String(description="Version code", example=__version__),
        })

        @self.api_namespace.route('/')
        class Version(Resource):
            @self.api.marshal_with(self.models["version"], code=200, description='OK')
            def get(self):
                """Check the CBR status, returning its version and configuration"""
                return {"version": __version__}

        # TODO: Add case structure marshall with it

        self.models["cases"] = self.api.model('Cases', {
            'cases': fields.List(fields.Raw, description="Cases", example=[("a", "b")]),
        })

        @self.api_namespace.route('/cases/')
        class Cases(Resource):
            # @self.api.marshal_with(self.models["cases"], code=200, description='OK')
            def get(self):
                """Check the cases in the case base"""
                return {"cases": case_base.to_json(orient="records")}

            def post(self):
                """Add a case to the case base (non-idempotent operation)"""
                raise NotImplementedError

        @self.api_namespace.route('/cases/<string:case_id>')
        class Case(Resource):
            # @self.api.marshal_with(self.models["cases"], code=200, description='OK')
            # @self.api.expect(models["retrieve"])
            def get(self, case_id):
                """Check a case in the case base"""
                raise NotImplementedError

            def put(self, case_id):
                """Add or update a case in the case base"""
                raise NotImplementedError

            def delete(self, case_id):
                """Check a case in the case base"""
                raise NotImplementedError

        # Note the patch in the example, which is due to pd.DataFrame.to_dict returning numpy types
        # Cf. https://github.com/pandas-dev/pandas/issues/16048
        self.models["retrieve"] = self.api.model('Retrieval', {
            'case': fields.Raw(description="Case", example=json.loads(case_base.iloc[0].to_json())),
            'k': fields.Integer(description="Number of neighbours", example=5)
        })

        @self.api_namespace.route('/retrieve/')
        class Retrieve(Resource):
            # @self.api.marshal_with(self.models["cases"], code=200, description='OK')
            @self.api.expect(models["retrieve"])
            def post(self):
                """Retrieve the most similar cases"""
                case = request.json.get("case")
                k = request.json.get("k", 5)
                df_sim, sims = recovery.find(pd.DataFrame([case]), k)[0]
                return {"cases": [row.dropna().to_dict() for _, row in df_sim.iterrows()],
                        "cases_ids": [i for i, _ in df_sim.iterrows()],
                        "sims": sims.tolist()}

        @self.api_namespace.route('/recommend/')
        class Recommend(Resource):
            # @self.api.marshal_with(self.models["cases"], code=200, description='OK')
            @self.api.expect(models["retrieve"])
            def post(self):
                """Provide a recommendation using the most similar cases"""
                case = request.json.get("case")
                k = request.json.get("k", 5)
                df_sim, sims = recovery.find(pd.DataFrame([case]), k)[0]
                return {"recommendation": aggregator.aggregate(df_sim, sims)}
