"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas_crosstabs.semantics import (
    add_semantics,
    return_semantics,
    copy_df,
    AXIS_NAMES,
)


def crosstab(
    df,
    row_fields,
    column_fields,
    ignore_nan=False,
    totals=None,
    totals_col=True,
    totals_row=True,
    totals_name='Totals',
):
    """
    Create frequency crosstab for selected categories.

    Parameters
    ==========
    :param df: DataFrame
    :param row_fields: str, list (of strings)
        Name(s) of DataFrame field(s) to add into the rows.
    :param column_fields: str, list (of strings), None
        Name(s) of DataFrame field(s) to add into the columns.

    Optional keyword arguments
    ==========================
    :param ignore_nan: boolean, default False
        Ignore category combinations if they have nans.
    :param totals_name: str, default 'Totals'
        Name for total rows/columns (string).
    :param totals: boolean, None default None
        Shorthand for setting `totals_col` and `totals_row`.
        Adds both totals to columns and to rows.
        If set, overrides both other arguments.
    :param totals_col: boolean, default True
        Add totals to columns.
    :param totals_row: boolean, default True
        Add totals to rows.

    Returns
    =======
    :crosstab: DataFrame
    """

    df_out = df.copy()

    if totals is not None:
        totals_col = totals
        totals_row = totals

    if not column_fields:
        column_fields = '_tmp'
        df_out[column_fields] = '_tmp'

    # assure row and column fields are lists
    if not isinstance(row_fields, list):
        row_fields = [row_fields]
    if not isinstance(column_fields, list):
        column_fields = [column_fields]

    margins = totals_col or totals_row

    # set columns to use/select from df_out
    group_cols = column_fields.copy()
    group_cols.extend(row_fields)

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
        index=row_fields,
        columns=column_fields,
        aggfunc='sum',
        dropna=False,
        margins=margins,
        margins_name=totals_name,
    )
    df_out = df_out.dropna(how='all')

    if margins:
        if not totals_col:
            df_out = df_out.drop(totals_name, axis=1, level=1)
        if not totals_row:
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
    if totals_col:
        df_out.semantics.col[-1] = 'T'
    if totals_row:
        df_out.semantics.row[-1] = 'T'
    return df_out


def percentages(
    df,
    axis='grand',
    label_abs='abs',
    labels_rel='%',
    roundings=1,
    level_name=None,
    totals='auto',
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

    if not isinstance(label_abs, str):
        raise TypeError('Label for the absolute columns has to be a string.')
    if not isinstance(axis, list):
        axis = [axis]
    if not isinstance(labels_rel, list):
        labels_rel = [labels_rel] * len(axis)
    if not len(axis) == len(labels_rel):
        raise IndexError(
            'Number of labels does not match the number '
            'of aggregation functions.'
            )
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
    label_abs='abs',
    label_rel='%',
    round=1,
    level_name=None,
    totals='auto',
):
    """
    Main logic to the `percentages` function.
    """

    valid_semantics = ['v', 't', 'T', ]

    # set percentage type
    perc_types = {
        0: 'r',
        1: 'c',
        'index': 'r',
        'columns': 'r',
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


def order_cat(df, categories):
    """
    Add ordered categories to a DataFrame.
    """

    categories = CategoricalDtype(categories=categories, ordered=True)
    df = df.astype(categories)
    return df
