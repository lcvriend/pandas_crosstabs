"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import numpy as np
import pandas as pd
from pandas_crosstabs.semantics import (
    return_semantics,
    copy_df,
    AXIS_NAMES,
)


def aggregations(
    df,
    level,
    axis=0,
    aggs='sum',
    labels=None,
    roundings=1
):
    """
    Aggregate within the specified level of a multiindex.
    Only value columns/rows will be used to perform the aggregation.

    Parameters
    ==========
    :param df: DataFrame
    :param level: int
        Level of the multiindex to be used.

    Optional keyword arguments
    ==========================
    :param axis: {0, 'index' or 'rows', 1 or 'columns'}, default 0
        If 0, 'index' or 'rows': apply function to row index.
        If 1 or 'columns': apply function to column index.
    :param aggs: {function, list or str}, default 'sum'
        Aggregation(s) to perform.
        - 'sum': sum of values
        - 'count': number of values
        - 'median': median
        - 'mean': mean
        - 'std': standard deviation
        - 'var': variance
        - 'min': minimum value
        - 'max': maximum value
        - func: function that aggregates a series and returns a scalar.
    :param labels: {list, str or None}, default None
        Label(s) for the aggregation row/column.
        If None will use the string or function name that is passed to `agg`.
    :param roundings: {list, int or None}, default 1
        Number of decimal places to round aggregation to.
        If None aggregation will not be rounded.

    Returns
    =======
    :aggregations: DataFrame
    """

    axis = AXIS_NAMES[axis]
    if not isinstance(aggs, list):
        aggs = [aggs]
    if labels is None:
        labels = [None] * len(aggs)
    if not isinstance(labels, list):
        labels = [labels]
    if not len(aggs) == len(labels):
        raise IndexError(
            'Number of labels does not match the number '
            'of aggregation functions.'
            )
    if isinstance(roundings, list):
        if not len(aggs) == len(roundings):
            raise IndexError(
                'Not every aggregation has a rounding decimal specified. '
                'Set list element to None if no rounding should be applied.'
                )
    else:
        roundings = [roundings] * len(aggs)

    if level == 0 or df.columns.nlevels > 1:
        aggs.reverse()
        labels.reverse()
        roundings.reverse()

    for agg, label, rounding in zip(aggs, labels, roundings):
        df = _add_agg(
            df, level, axis=axis, agg=agg, label=label, round=rounding
            )
    return df


def _add_agg(df, level, axis=0, agg='sum', label=None, round=1):
    """
    Main logic to the add_sub_agg function.
    """

    valid_semantics = ['v', 'T', 't']

    semantic_code = {
        'sum': 't',
        'count': 'c',
        }
    semantic_code = semantic_code.get(agg, 'a')

    if label is None:
        if isinstance(agg, str):
            label = agg
        else:
            label = agg.__name__

    original_dtypes = dict(zip(df.columns.values, df.dtypes.values))
    df_out = copy_df(df, transpose=not axis)
    df = df_out.copy()
    col_semantics = df_out.semantics.col
    row_semantics = df_out.semantics.row
    nlevels = df.columns.nlevels
    valid_rows = [item in valid_semantics for item in row_semantics]

    # DataFrame without MultiIndex
    if nlevels == 1:
        agg_vals = return_semantics(df_out, axis=1).agg(agg, axis=1)
        if round is not None:
            agg_vals = agg_vals.round(round)
        df_out.loc[valid_rows, label] = agg_vals
        df_out.semantics.col.append('a')
        if axis == 0:
            df_out = df_out.T
            df_out = df_out.astype(original_dtypes)
            df_out.semantics.col = row_semantics
            df_out.semantics.row = col_semantics
        else:
            df_out.semantics.col = col_semantics
            df_out.semantics.row = row_semantics
        return df_out

    # DataFrame with MultiIndex
    # set levels
    if level > nlevels - 1:
        raise Exception(
            f'The index has {nlevels} useable levels: '
            f'{list(range(nlevels))}. Level {level} is out of bounds.'
            )

    # deal with categorical indexes
    if df_out.columns.levels[level].dtype.name == 'category':
        new_level = df_out.columns.levels[level].add_categories(label)
        df_out.columns.set_levels(new_level, level=level, inplace=True)

    i = level
    while i < (nlevels - 1):
        try:
            new_level = df_out.columns.levels[i].add_categories('')
            df_out.columns.set_levels(new_level, level=i, inplace=True)
        except AttributeError:
            pass
        i += 1

    # collect columns with values and columns with content
    content = ['v', 'pg', 'pr', 'pc']
    v_cols = [elem == 'v' for elem in col_semantics]
    c_cols = [elem in content for elem in col_semantics]

    # aggregation on level == 0
    if level == 0:
        empty = ('',) * (nlevels - 1)
        key = label, *empty
        agg_vals = df.loc[valid_rows, v_cols].agg(agg, axis=1)
        if round is not None:
            agg_vals = agg_vals.round(round)
        df_out[key] = agg_vals
        df_out.semantics.col.append('a')

    # aggregation on level > 0
    else:
        # collect column keys for specified level
        col_keys = list()
        for col in df.loc[:, v_cols].columns.remove_unused_levels():
            fnd_col = col[: level]
            col_keys.append(fnd_col)
        col_keys = list(dict.fromkeys(col_keys))

        # select groups from table, sum them and add to df
        level_list = list(range(level))
        for key in col_keys:
            # find last key in group
            tbl_grp = (
                df.loc[:, c_cols]
                .xs([*key], axis=1, level=level_list, drop_level=False)
                )
            key_last_col = tbl_grp.iloc[:, -1].name

            # set new column key
            lst_last_col = list(key_last_col)
            lst_last_col[level] = label
            i = level + 1
            while i < (nlevels):
                lst_last_col[i] = ''
                i += 1
            key_new_col = tuple(lst_last_col)

            # insert new column
            idx_col = df_out.columns.get_loc(key_last_col) + 1
            df_out.insert(idx_col, key_new_col, np.nan)
            col_semantics.insert(idx_col, semantic_code)

            # aggregate and update
            tbl_grp = (
                df.loc[:, v_cols]
                .xs([*key], axis=1, level=level_list, drop_level=False)
                )

            agg_vals = tbl_grp.agg(agg, axis=1).values
            if round is not None:
                agg_vals = agg_vals.round(round)
            multiindex = pd.MultiIndex.from_tuples([key_new_col])
            df_col = pd.DataFrame(
                data=agg_vals,
                columns=multiindex,
                index=df.index,
                )
            df_out.update(df_col.loc[valid_rows, :])

    # transpose if aggregating on index and store semantics
    if axis == 0:
        df_out = df_out.T
        for col in df_out.columns:
            try:
                df_out[col] = df_out[col].astype(original_dtypes[col])
            except ValueError:
                df_out[col] = df_out[col].astype('float')
        df_out.semantics.col = row_semantics
        df_out.semantics.row = col_semantics
    else:
        df_out.semantics.col = col_semantics
        df_out.semantics.row = row_semantics
    return df_out
