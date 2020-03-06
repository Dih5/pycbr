"""
Module providing the functionality to define case bases
"""

import pandas as pd


class CaseBase:
    def __init__(self):
        pass

    def get_pandas(self):
        """
        Get a pandas dataframe representing the case base

        Returns:
            pandas.DataFrame: A dataframe representing the case base.

        """
        raise NotImplementedError

    def add_case(self, case, case_id=None):
        """
        Add a case to the case base

        Args:
            case: A description of the case.
            case_id: An identifier of the case.

        """
        raise NotImplementedError

    def delete_case(self, case_id):
        """
        Remove a case from the case base

        Args:
            case_id: Unique identifier of the case

        """
        raise NotImplementedError

    def get_description(self):
        """Get a dictionary describing the instance"""
        raise NotImplementedError


class SimpleCSVCaseBase(CaseBase):
    """A CSV file storing the case base with no synchronization options"""

    def __init__(self, path, csv_kwargs=None):
        """

        Args:
            path (str): Location of the CSV with the case base.
            csv_kwargs (dict): Additional parameters describing the CSV file.
        """
        super().__init__()
        self.path = path
        self.csv_kwargs = csv_kwargs if csv_kwargs is not None else {}
        # TODO: Handle index presence in the file

        self.df = None
        self.df = self.get_pandas()
        self.header = list(self.df.columns)

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "path": self.path, "csv_kwargs": self.csv_kwargs}

    def get_pandas(self):
        # self.df is assumed to be up-to-date with the CSV
        if self.df is not None:
            return self.df
        else:
            self.df = pd.read_csv(self.path, **self.csv_kwargs)
            return self.df

    def add_case(self, case, case_id=None):
        df = self.get_pandas()
        if case_id is None:
            self.df = df.append(case, ignore_index=True)
            # Note the last row might contain line breaks (inside a quote delimiter).
            with open(self.path, "a") as f:
                self.df.iloc[[-1]].to_csv(f, header=None, index=False,
                                          **{k: v for k, v in self.csv_kwargs.items() if k != "header"})
        else:
            # If index is not stored, new ids will be meaningless when reloading the file
            if case_id not in df.index:
                raise NotImplementedError("SimpleCSVCaseBase does not allow adding a new case with a chosen id")
            df2 = pd.DataFrame([case], columns=self.header)
            self.df.loc[case_id] = df2.iloc[0]
            self.df.to_csv(self.path, index=False, **self.csv_kwargs)

    def delete_case(self, case_id):
        self.df.drop(case_id, inplace=True)
        self.df.to_csv(self.path, index=False, **self.csv_kwargs)
