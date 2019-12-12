"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import pandas as pd
from pandas_crosstabs.config import PERCENTAGES
from pandas_crosstabs.semantics import (
    add_semantics,
    return_semantics,
    copy_df,
)


def percentages_old(
    df,
    axis='grand',
    roundings=1,
    level_name=None,
    totals='auto',
    **kwargs,
):
    """
    Add percentage columns to a DataFrame.
    Ignores aggregation columns (such as mean, std, etc.)

    Parameters
    ==========
    :param df: DataFrame

    Optional keyword arguments
    ==========================
    :param axis: {list, 'grand', 'index' or 0, 'columns' or 1}, default 'grand'
        'grand' - Calculate percentages from grand total.
        'index', 0 - Calculate percentages from row totals.
        'columns', 1 - Calculate percentages from column totals.
    :param label_abs: string, default 'abs'
        Label for the absolute columns.
    :param labels_rel: {list, string}, default '%'
        Label(s) for the relative columns.
    :param roundings: {list, int or None}, default 1
        Number of decimal places to round percentages to.
        If None percentages will not be rounded.
    :param level_name: {str, None}, default None
        Name of the level added to the columns.
    :param totals: boolean, {'auto'}, default 'auto'
        'auto' - Check automatically (may backfire).
        True - Take the totals from the DataFrame (last row/column/value).
        False - Calculate the totals.

    Returns
    =======
    :percentages: DataFrame
    """

    if isinstance(roundings, list):
        if not len(axis) == len(roundings):
            raise IndexError(
                'Not every aggregation has a rounding decimal specified. '
                'Set list element to None if no rounding should be applied.'
                )
    else:
        roundings = [roundings] * len(axis)

    axis.reverse()
    labels_rel.reverse()
    roundings.reverse()

    for axis, label_rel, rounding in zip(axis, labels_rel, roundings):
        df = _add_per_col(
            df,
            axis=axis,
            label_abs=label_abs,
            label_rel=label_rel,
            round=rounding,
            level_name=level_name,
            totals=totals,
            )
    return df


def _add_per_col(
    df,
    axis='grand',
    label_abs=PERCENTAGES.label_absolute,
    label_rel=PERCENTAGES.label_relative,
    round=1,
    level_name=None,
    totals='auto',
):
    """
    Main logic to the `percentages` function.
    """

    if not isinstance(label_abs, str):
        raise TypeError('Label for the absolute columns has to be a string.')
    if not isinstance(axis, list):
        axis = [axis]
    if not isinstance(label_rel, list):
        labels_rel = [label_rel] * len(axis)
    if not len(axis) == len(label_rel):
        raise IndexError(
            'Number of labels does not match the number '
            'of aggregation functions.'
            )

    valid_semantics = ['v', 't', 'T', ]

    # set percentage type
    perc_types = {
        0: 'r',
        1: 'c',
        'index': 'r',
        'columns': 'c',
        'grand': 'g',
        }
    type_names = {
        'r': 'rows',
        'c': 'cols',
        'g': 'grnd',
    }
    if axis not in perc_types:
        raise KeyError(
            f'Unexpected input for axis. Valid values are: {perc_types.keys()}'
            )
    perc_type = perc_types[axis]

    df_out = copy_df(df)
    df_out = add_semantics(df_out)
    col_semantics = df_out.semantics.col.copy()
    row_semantics = df_out.semantics.row.copy()
    valid_rows = [item in valid_semantics for item in row_semantics]

    # add column index for labelling percentage columns
    edit = True
    if not any([item.lower().startswith('p') for item in col_semantics]):
        edit = False
        nlevels = df_out.columns.nlevels + 1
        levels = list(range(nlevels))
        levels.append(levels.pop(0))
        lvl_colnames = [
            label_abs if item in valid_semantics else ''
            for item in df_out.semantics.col
            ]
        df_out = pd.concat(
                [df_out], axis=1, keys=[label_abs],
            ).reorder_levels(
                levels, axis=1,
            )
        new_cols = list()
        new_colnames = list(df_out.columns.names)
        new_colnames[-1] = level_name
        for idx, col in enumerate(df.columns):
            tup = col, lvl_colnames[idx]
            if isinstance(col, tuple):
                tup = *col, lvl_colnames[idx]
            new_cols.append(tup)
        new_cols = pd.MultiIndex.from_tuples(new_cols, names=new_colnames)
        df_out.columns = new_cols

    # add percentage columns
    for idx, col in enumerate(df.columns):
        # skip column if percentage does not make sense for data type
        if df.semantics.col[idx] not in valid_semantics:
            continue

        # skip column if type is already present within next three columns
        try:
            for i in [1, 2, 3, ]:
                item = df.semantics.col[idx + i].lower()
                # break if next column is not percentage column
                if not item.startswith('p'):
                    check = False
                    break
                check = item == f'p{perc_type}'
                if check:
                    break
            if check:
                continue
        except IndexError:
            pass

        # set column names
        if edit:
            new_col = *col[:-1], label_rel
            abs_col = col
        else:
            new_col = col, label_rel
            abs_col = col, label_abs
            if isinstance(col, tuple):
                new_col = *col, label_rel
                abs_col = *col, label_abs

        # test uniqueness of new column
        if new_col in [col for col in df_out.columns]:
            new_col = *col[:-1], f'{label_rel}-{type_names[perc_type]}'

        # calculate and add percentages
        total = _find_total(df, col, axis, totals)
        col_idx = df_out.columns.get_loc(abs_col)
        new_cols = df_out.columns.insert(col_idx + 1, new_col)
        col_semantics.insert(col_idx + 1, f'p{perc_type}')
        if col_semantics[col_idx] == 'T':
            col_semantics[col_idx + 1] = f'P{perc_type}'
        df_out = pd.DataFrame(df_out, columns=new_cols)
        df_out[new_col] = df_out.loc[valid_rows, abs_col] / total * 100
        if round is not None:
            df_out[new_col] = df_out[new_col].round(round)

    df_out.semantics.col = col_semantics
    df_out.semantics.row = row_semantics
    return df_out


def _find_total(df, col, axis, totals):
    """
    Mapper for mapping the totals functions.
    """

    mapper = {
        0: _totals_row,
        1: _totals_col,
        'index': _totals_row,
        'columns': _totals_col,
        'grand': _grand_total,
        }
    if axis not in [1, 'columns']:
        total = mapper[axis](df, totals)
    else:
        total = mapper[axis](df, col, totals)
    return total


def _totals_col(df, col, totals):
    """
    Return the total of a column in a DataFrame.
    First check if the column already has a column total.
    If not, caclulate the total. Return the total.

    :returns: int, float
    """

    nrows, _ = df.shape
    try:
        df = return_semantics(df, axis=0, semantics=['v', 'T'])
        if 'T' in df.semantics.row:
            sel = [item == 'T' for item in df.semantics.row]
            return df.iloc[sel][col].values[0]
        else:
            return df[col].sum()
    except TypeError:
        total = df.iloc[-1][col]
        if totals == 'auto':
            if not total == df.iloc[:nrows - 1][col].sum():
                total = df[col].sum()
            return total
        elif totals:
            return total
        else:
            return df[col].sum()


def _totals_row(df, totals):
    """
    Return row totals of a DataFrame as a Series.
    First check if the DataFrame already has a column with row totals.
    If not calculate it. Return the Series with row totals.

    :returns: Series
    """

    _, ncols = df.shape
    try:
        df = return_semantics(df, axis=1, semantics=['v', 'T'])
        if 'T' in df.semantics.col:
            sel = [item == 'T' for item in df.semantics.col]
            df = df.loc[:, sel]
            return df.iloc[:, 0]
        else:
            return df.sum(axis=1)
    except TypeError:
        total = df.iloc[:, -1]
        if totals == 'auto':
            if not total.equals(df.iloc[:, :ncols - 1].sum(axis=1)):
                total = df.sum(axis=1)
            return total
        elif totals:
            return total
        else:
            return df.sum(axis=1)


def _grand_total(df, totals):
    """
    Return the grand total of a DataFrame.
    First check if the DataFrame already has a grand total.
    If not calculate it. Return the grand total.

    :returns: int, float
    """

    nrows, ncols = df.shape
    try:
        df = return_semantics(df, semantics=['v', 'T'])
        total_col = 'T' in df.semantics.col
        total_row = 'T' in df.semantics.row
        sel_col = [item == 'T' for item in df.semantics.col]
        sel_row = [item == 'T' for item in df.semantics.row]
        if total_col and total_row:
            return df.loc[sel_row, sel_col].values[0][0]
        elif total_col:
            return df.loc[:, sel_col].sum().values[0]
        elif total_row:
            return df.loc[sel_row, :].sum(axis=1).values[0]
        else:
            return df.sum().sum()
    except TypeError:
        total = df.iloc[-1, -1]
        if totals == 'auto':
            if not total == df.iloc[:nrows - 1, :ncols - 1].values.sum():
                total = _totals_row(df, 'auto').values.sum()
            return total
        elif totals:
            return total
        else:
            return _totals_row(df, False).values.sum()
