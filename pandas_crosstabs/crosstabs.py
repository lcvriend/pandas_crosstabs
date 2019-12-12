"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas_crosstabs.semantics import (
    add_semantics,
)


def crosstab(
    df,
    index,
    columns=None,
    ignore_nan=False,
    totals=None,
    totals_columns=True,
    totals_index=True,
    totals_name='Totals',
):
    """
    Create frequency crosstab for selected categories.

    Parameters
    ==========
    :param df: DataFrame
    :param index: str, list (of strings)
        Name(s) of DataFrame field(s) to add into the rows.
    :param columns: str, list (of strings), None
        Name(s) of DataFrame field(s) to add into the columns.

    Optional keyword arguments
    ==========================
    :param ignore_nan: boolean, default False
        Ignore category combinations if they have nans.
    :param totals_name: str, default 'Totals'
        Name for total rows/columns (string).
    :param totals: boolean, None default None
        Shorthand for setting `totals_columns` and `totals_index`.
        Adds both totals to columns and to rows.
        If set, overrides both other arguments.
    :param totals_columns: boolean, default True
        Add totals to columns.
    :param totals_index: boolean, default True
        Add totals to rows.

    Returns
    =======
    :crosstab: DataFrame
    """

    df_out = df.copy()

    if totals is not None:
        totals_columns = totals
        totals_index = totals

    if not columns:
        columns = '_tmp'
        df_out[columns] = '_tmp'

    # assure row and column fields are lists
    if not isinstance(index, list):
        index = [index]
    if not isinstance(columns, list):
        columns = [columns]

    margins = totals_columns or totals_index

    # set columns to use/select from df_out
    group_cols = columns.copy()
    group_cols.extend(index)

    # fill nan if ignore_nan is False
    if not ignore_nan:
        for col in group_cols:
            if df_out[col].isnull().values.any():
                if df_out[col].dtype.name == 'category':
                    df_out[col] = df_out[col].cat.add_categories([''])
                df_out[col] = df_out[col].fillna('')

    # find column for counting that is not in group_cols
    check = False
    i = 0
    while not check:
        try:
            col = df_out.columns[i]
            if col not in group_cols:
                check = True
            i += 1
        except KeyError:
            df_out['_tmp'] = '_tmp'
            col = '_tmp'
            check = True

    # pivot table
    df_out = df_out.groupby(group_cols)[[col]].count()
    df_out = df_out.dropna()
    df_out = pd.pivot_table(
        df_out.reset_index(),
        index=index,
        columns=columns,
        aggfunc='sum',
        dropna=False,
        margins=margins,
        margins_name=totals_name,
    )
    df_out = df_out.dropna(how='all')

    if margins:
        if not totals_columns:
            df_out = df_out.drop(totals_name, axis=1, level=1)
        if not totals_index:
            try:
                df_out = df_out.drop(totals_name, axis=0, level=0)
            except KeyError:
                df_out = df_out.drop(totals_name, axis=0)
    df_out.columns = df_out.columns.droplevel(0)
    df_out = df_out.fillna(0)

    # remove row/columns where all values are 0
    df_out = df_out.loc[(df_out != 0).any(axis=1)]
    df_out = df_out.loc[:, (df_out != 0).any(axis=0)]
    df_out = df_out.astype('int64')

    # try to remove temp column
    try:
        df_out = df_out.drop('_tmp', axis=1)
        df_out.columns.name = ''
    except KeyError:
        pass

    # add semantics
    df_out = add_semantics(df_out)
    if totals_columns:
        df_out.semantics.col[-1] = 'T'
    if totals_index:
        df_out.semantics.row[-1] = 'T'
    return df_out


def order_cat(df, categories):
    """
    Add ordered categories to a DataFrame.
    """

    categories = CategoricalDtype(categories=categories, ordered=True)
    df = df.astype(categories)
    return df
