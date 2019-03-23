"""
Module for adding semantics to DataFrame.
The semantics attribute of a DataFrame stores the column/row types.
"""

import itertools
import pandas as pd


@pd.api.extensions.register_dataframe_accessor('semantics')
class Semantics(object):
    SEMANTIC_VALUES = {
        'v': 'value',
        'T': 'total',
        't': 'subtotal',
        'Pg': 'percentage grand total',
        'Pr': 'percentage row total',
        'Pc': 'percentage column total',
        'pg': 'percentage grand value',
        'pr': 'percentage row value',
        'pc': 'percentage column value',
        'c': 'count',
        'a': 'aggregation',
    }

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._col = None
        self._row = None

    def __repr__(self):
        representation = (
            f'cols: {" | ".join(self.col)}\n'
            f'rows: {" | ".join(self.row)}'
        )
        return representation

    @property
    def col(self):
        return self._col

    @property
    def row(self):
        return self._row

    @col.setter
    def col(self, col):
        self._col = col

    @row.setter
    def row(self, row):
        self._row = row


def add_semantics(df):
    """
    Add basic semantics to DataFrame if not already present.
    """

    if df.semantics.col is None:
        df.semantics.col = df.shape[1] * ['v']
    if df.semantics.row is None:
        df.semantics.row = df.shape[0] * ['v']
    return df


def return_semantics(df, axis='both', semantics='v'):
    """
    Return DataFrame with only the specified semantics.
    """

    if isinstance(semantics, str):
        semantics = [semantics]
    if not isinstance(semantics, list):
        raise TypeError(
            f'Semantics argument must be passed a string '
            f'or a list of strings representing the possible semantic values: '
            f'{Semantics.SEMANTIC_VALUES}'
            )

    cols = [item in semantics for item in df.semantics.col]
    rows = [item in semantics for item in df.semantics.row]
    col_semantics = list(itertools.compress(df.semantics.col, cols))
    row_semantics = list(itertools.compress(df.semantics.row, rows))
    if not axis == 'both':
        axis = AXIS_NAMES[axis]
        if axis == 0:
            df_out = df.loc[rows, :]
            df_out.semantics.row = row_semantics
            df_out.semantics.col = df.semantics.col
            return df_out
        else:
            df_out = df.loc[:, cols]
            df_out.semantics.col = col_semantics
            df_out.semantics.row = df.semantics.row
            return df_out
    df_out = df.loc[rows, cols]
    df_out.semantics.col = col_semantics
    df_out.semantics.row = row_semantics
    return df_out


def copy_df(df, transpose=False):
    """
    Copy DataFrame while preserving the semantics.
    """

    df = add_semantics(df)
    col_semantics = df.semantics.col.copy()
    row_semantics = df.semantics.row.copy()

    if not transpose:
        df = df.copy()
        df.semantics.col = col_semantics
        df.semantics.row = row_semantics
    else:
        df = df.T.copy()
        df.semantics.col = row_semantics
        df.semantics.row = col_semantics
    return df


AXIS_NAMES = {
    0: 0,
    1: 1,
    'index': 0,
    'rows': 0,
    'columns': 1,
}
