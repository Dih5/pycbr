"""pycbr - Package to implement Case-Based Reasoning systems"""

__version__ = '0.1.1'
__author__ = 'Dih5 <dihedralfive@gmail.com>'

from . import aggregate, models, recovery, casebase, server


class CBR:
    """A CBR application"""

    def __init__(self, case_base, recovery_model, aggregator=None, refit_always=True, create_server=True,
                 server_name="pycbr"):
        """

        Args:
            case_base (casebase.CaseBase): An instance defining how the case base persistence is handled.
            recovery_model (recovery.Recovery): Instance defining how similarity is measured and how the search is
                                                performed.
            aggregator (aggregate.Aggregate): Aggregation procedure to propose a solution from a set of cases.
            refit_always (bool): Whether to refit the similarities whenever a case is modified. Might be deactivated
                                 for large case bases for performance reasons.
            create_server (bool): Whether to create the Flask WSGI app.
            server_name (str): Name to assign to the server.

        """
        self.case_base = case_base
        self.recovery_model = recovery_model
        self.aggregator = aggregator
        self.refit_always = refit_always

        self._int_index = None

        self.refit()

        if create_server:
            self.server = server.CBRFlask(server_name, self)
            self.app = self.server.app
        else:
            self.server = None
            self.app = None

    def refit(self):
        """Update the recovery model to match the case base"""
        df = self.case_base.get_pandas()
        self._int_index = df.index.dtype in ["int16", "int32", "int64"]
        self.recovery_model.fit(df)

    def get_pandas(self):
        """
        Get a pandas dataframe representing the case base

        Returns:
            pandas.DataFrame: A dataframe representing the case base.

        """
        return self.case_base.get_pandas()

    def get_case(self, case_id):
        """
        Retrieve a case with the given id

        Args:
            case_id: An identifier of the id.

        Returns:
            pandas.Series: The case found.

        """
        df = self.get_pandas()
        if self._int_index:
            case_id = int(case_id)
        return df.loc[case_id]

    def add_case(self, case, case_id=None):
        """
        Add a case to the case base

        Args:
            case: A description of the case.
            case_id: An identifier of the case.

        """
        if case_id is not None and self._int_index:
            case_id = int(case_id)
        r = self.case_base.add_case(case, case_id=case_id)
        if self.refit_always:
            self.refit()
        return r

    def delete_case(self, case_id):
        """
        Remove a case from the case base

        Args:
            case_id: Unique identifier of the case

        """
        if self._int_index:
            case_id = int(case_id)
        r = self.case_base.delete_case(case_id)
        if self.refit_always:
            self.refit()
        return r
