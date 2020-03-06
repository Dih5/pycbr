"""
Module providing the functionality to build Flask WSGI applications for the CBR
"""
import logging
import logging.config
import os
import json

import coloredlogs
import yaml
import pandas as pd

from flask import Flask, request, abort
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


def _pandas_to_python(instance):
    """Convert a pandas object to a standard python object"""
    # This patch is is due to pd.DataFrame.to_dict returning numpy types
    # Cf. https://github.com/pandas-dev/pandas/issues/16048
    return json.loads(instance.to_json())


class CBRFlask:
    def __init__(self, import_name, cbr):
        self.app = Flask(import_name)
        CORS(self.app)

        self.api = Api(app=self.app, version=__version__, title=import_name, description="A pyCBR generated CBR API")
        self.api_namespace = self.api.namespace("api", description="General methods")

        self.app.config.update(
            ERROR_404_HELP=False,  # No "but did you mean" messages
            RESTPLUS_MASK_SWAGGER=False,
        )

        # Log POST bodies
        @self.app.before_request
        def log_request_info():
            # self.app.logger.debug('Headers: %s', request.headers)
            data = request.get_data()
            if data:
                self.app.logger.info('Body: %s', data.decode())

        self.case_base = cbr.case_base

        self.models = {}
        models = self.models

        self.models["version"] = self.api.model('Deployment configuration', {
            'version': fields.String(description="pycbr version code", example=__version__),
            'case_base': fields.Raw(description="Case base configuration"),
            'recovery_model': fields.Raw(description="Recovery model"),
            'aggregator': fields.Raw(description="Aggregation mechanism")
        })

        @self.api_namespace.route('/')
        class Version(Resource):
            @self.api.marshal_with(self.models["version"], code=200, description='OK')
            def get(self):
                """Check the CBR status, returning its version and configuration"""
                return {"version": __version__, "case_base": cbr.case_base.get_description(),
                        "recovery_model": cbr.recovery_model.get_description(),
                        "aggregator": cbr.aggregator.get_description() if cbr.aggregator is not None else None
                        }

        case_example = _pandas_to_python(cbr.get_pandas().iloc[0])

        self.models["case"] = self.api.model('Case', {k: fields.Raw(example=v) for k, v in case_example.items()})

        # TODO: List of cases marshalling
        # self.models["cases"] = self.api.model('Cases', {'cases': fields.List(fields.Nested(self.models["case"]), description="Cases", example=[case_example])})

        @self.api_namespace.route('/cases/')
        class Cases(Resource):
            # @self.api.marshal_with(self.models["cases"], code=200, description='OK')
            def get(self):
                """Check the cases in the case base"""
                return {"cases": _pandas_to_python(cbr.get_pandas())}

            @self.api.expect(models["case"])
            def post(self):
                """Add a case to the case base (non-idempotent operation)"""
                cbr.add_case(request.json)

        @self.api_namespace.route('/cases/<string:case_id>')
        class Case(Resource):
            # @self.api.marshal_with(self.models["case"], code=200, description='OK')
            def get(self, case_id):
                """Check a case in the case base"""
                return _pandas_to_python(cbr.get_case(case_id))

            @self.api.expect(models["case"])
            def put(self, case_id):
                """Add or update a case in the case base"""
                cbr.add_case(request.json, case_id=case_id)

            def delete(self, case_id):
                """Check a case in the case base"""
                try:
                    cbr.delete_case(case_id)
                except KeyError:
                    abort(404)

        self.models["retrieve"] = self.api.model('Retrieval', {
            'case': fields.Nested(self.models["case"], description="Case", example=case_example),
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
                df_sim, sims = cbr.recovery_model.find(pd.DataFrame([case]), k)[0]
                return {"cases": [row.dropna().to_dict() for _, row in df_sim.iterrows()],
                        "cases_ids": [i for i, _ in df_sim.iterrows()],
                        "sims": sims.tolist()}

        if cbr.aggregator is not None:
            @self.api_namespace.route('/recommend/')
            class Recommend(Resource):
                # @self.api.marshal_with(self.models["cases"], code=200, description='OK')
                @self.api.expect(models["retrieve"])
                def post(self):
                    """Provide a recommendation using the most similar cases"""
                    case = request.json.get("case")
                    k = request.json.get("k", 5)
                    df_sim, sims = cbr.recovery_model.find(pd.DataFrame([case]), k)[0]
                    return {"recommendation": cbr.aggregator.aggregate(df_sim, sims)}
