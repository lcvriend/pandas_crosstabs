"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import itertools
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path


class FancyTable:
    """
    Class for adding custom styling to DataFrames.
    """
    CLASS_VALUES_COL = {
        'p': 'perc_col',
        'T': 'totals_col',
        'P': 'totals_perc_col',
        'a': 'agg_col',
    }
    CLASS_VALUES_ROW = {
        'T': 'totals_row',
        'a': 'agg_row',
    }

    path_to_css = Path(__file__).resolve().parent
    tab_len = 4
    tab = ' ' * tab_len

    def __init__(
        self,
        df,
        style=None,
        edge_lvl_row=None,
        edge_lvl_col=None,
    ):
        self.df = df
        self.nlevels_col = df.columns.nlevels
        self.nlevels_row = df.index.nlevels
        self.nrows, self.ncols = df.shape
        self.styling = Style(style)
        self.style = self.styling.style
        self.css = self.styling.css

        self._edge_lvl_row = (
            self._lvl_find_value(edge_lvl_row, 'row')
            if edge_lvl_row is not None
            else self._lvl_find_value(self.styling.edge_lvl_row, 'row')
            )
        self._edge_lvl_col = (
            self._lvl_find_value(edge_lvl_col, 'col')
            if edge_lvl_col is not None
            else self._lvl_find_value(self.styling.edge_lvl_col, 'col')
            )

        self.row_edges = self.find_edges(
            self.df.index,
            self.nlevels_row,
            edge_level=self.edge_lvl_row,
            )
        self.col_edges = self.find_edges(
            self.df.columns,
            self.nlevels_col,
            edge_level=self.edge_lvl_col,
            )

    @property
    def edge_lvl_row(self):
        return self._edge_lvl_row

    @edge_lvl_row.setter
    def edge_lvl_row(self, value):
        self._edge_lvl_row = self._lvl_find_value(value, 'row')

    @property
    def edge_lvl_col(self):
        return self._edge_lvl_col

    @edge_lvl_col.setter
    def edge_lvl_col(self, value):
        self._edge_lvl_col = self._lvl_find_value(value, 'col')

    def _lvl_find_value(self, value, axis):
        """
        Helper function for finding the correct edge level value.

        :Returns:
            `value` if `value` passed is integer
            -1 if `value` passed is None
            0 if `value` passed is 'nogrid'
            row/col `nlevels` if `value` passed is 'full'
            int if `value` is string and none of the above
        """

        if value is None:
            return -1
        if isinstance(value, str):
            if value == 'nogrid':
                value = 0
            elif value == 'full':
                if axis == 'row':
                    value = self.nlevels_row
                else:
                    value = self.nlevels_col
            else:
                value = int(value)
        return value

    @property
    def html(self):
        return self._html()

    def _repr_html_(self):
        return(f'<style>{self.css}</style>\n{self._html()}')

    def _html(self):
        html_tbl = (
            f'<table class="{self.style}">\n'
            f'{self._thead()}{self._tbody()}</table>\n'
        )
        return html_tbl

    def _thead(self):
        all_levels = list()

        # columns
        i = 0
        while i < self.nlevels_col:
            level = list()

            # column index
            col_idx_name = self.df.columns.get_level_values(i).name
            if col_idx_name is None:
                col_idx_name = ''
            colspan = ''
            if self.nlevels_row > 1:
                colspan = f' colspan="{self.nlevels_row}"'

            html_repr = (
                f'<th class="col_idx_name"{colspan}>{col_idx_name}</th>\n'
                )
            level.append(html_repr)

            # column names
            col_names = self.get_idx_keys(self.df.columns, i)
            spans = self.find_spans(col_names)
            html_repr = self.set_idx_names(spans, i, axis=1)
            level.extend(html_repr)
            level = [f'{self.tab * 3}{el}' for el in level]

            all_levels.append(level)
            i += 1

        # row index names
        def html_repr_idx_names(idx_name):
            if idx_name is None:
                idx_name = ''
            html_repr = f'<th class="row_idx_name">{idx_name}</td>\n'
            return html_repr

        idx_names = list(self.df.index.names)
        if not idx_names == [None] * len(idx_names):
            level = [html_repr_idx_names(idx_name) for idx_name in idx_names]

            def html_repr_idx_post(col_idx, item):
                classes = list()
                if col_idx in self.col_edges:
                    classes.append('col_edge')
                for key in self.CLASS_VALUES_COL:
                    try:
                        if self.df.semantics.col[col_idx] == key:
                            classes.append(self.CLASS_VALUES_COL[key])
                    except KeyError:
                        pass
                classes = ' '.join(classes)
                classes = f' {classes}' if classes != '' else ''
                html_repr = f'<td class="row_idx_post{classes}"></td>\n'
                return html_repr

            level.extend(
                [html_repr_idx_post(col_idx, item)
                for col_idx, item in enumerate([''] * self.ncols)]
                )
            level = [f'{self.tab * 3}{el}' for el in level]
            all_levels.append(level)

        # convert to html
        html = ''
        for level in all_levels:
            html += f'{self.tab * 2}<tr class="tbl_row">\n'
            html += ''.join(level)
            html += f'{self.tab * 2}</tr>\n'
        thead = f'{self.tab}<thead>\n{html}{self.tab}</thead>\n'
        return thead

    def _tbody(self, tid='cell'):
        row_elements = list()

        # indices
        i = 0
        while i < self.nlevels_row:
            idx_names = self.get_idx_keys(self.df.index, i)
            spans = self.find_spans(idx_names)
            level = self.set_idx_names(spans, i, axis=0)
            row_elements.append(level)
            i += 1

        # values
        def html_repr(col_idx, item):
            # find classes to add to col
            classes = list()
            if col_idx in self.col_edges:
                classes.append('col_edge')
            for key in self.CLASS_VALUES_COL:
                try:
                    if self.df.semantics.col[col_idx] == key:
                        classes.append(self.CLASS_VALUES_COL[key])
                except TypeError:
                    pass
            classes = ' '.join(classes)
            classes = f' {classes}' if classes != '' else ''

            # write html line
            html_repr = (
                f'<td id="{tid}-{row_idx + 1}-{col_idx + 1}" '
                f'class="tbl_cell{classes}">{item}</td>\n'
                )
            return html_repr

        values = self.df.astype(str).values  # cast all values as strings
        row_vals = list()
        for row_idx, row in enumerate(values):
            val_line = [
                html_repr(col_idx, item)
                for col_idx, item in enumerate(row)
                ]
            val_line = (self.tab * 3).join(val_line)
            row_vals.append(val_line)
        row_elements.append(row_vals)

        # zip indices and values
        rows = list(zip(*row_elements))

        # write tbody
        html = ''
        for idx, row in enumerate(rows):
            # find classes to add to row
            classes = list()
            if idx - 1 in self.row_edges:
                classes.append('row_edge')
            for key in self.CLASS_VALUES_ROW:
                try:
                    if self.df.semantics.row[idx] == key:
                        classes.append(self.CLASS_VALUES_ROW[key])
                except TypeError:
                    pass
            classes = ' '.join(classes)
            classes = f' {classes}' if classes != '' else ''

            # write html line
            row_str = (self.tab * 2) + f'<tr class="tbl_row{classes}">\n'
            row_str += ''.join(
                [(self.tab * 3) + item for item in row if item is not None]
                )
            row_str += (self.tab * 2) + '</tr>\n'
            html += row_str
        tbody = f'{self.tab}<tbody>\n{html}{self.tab}</tbody>\n'
        return tbody

    @staticmethod
    def get_idx_keys(index, level):
        """
        Get list of labels from index or list of sliced tuples from multiindex.
        """

        if index.nlevels > 1:
            return [key[:level + 1] for key in index]
        return [key for key in index]

    def set_idx_names(self, spans, level, axis=0):
        """
        Return a list of html strings containing the index names.
        Does something different depending on the axis.
        """

        types = {
            0: 'row',
            1: 'col'
        }
        prefix = types[axis]
        idx_names = list()

        spn_idx_left = 0
        spn_idx_right = -1
        for span in spans:
            idx_name = span[0]
            spn_idx_right += span[1]
            spn_str = ''
            classes = ''

            if span[1] > 1:
                spn_str = f' {prefix}span="{span[1]}"'
            if isinstance(span[0], tuple):
                idx_name = span[0][level]

            if axis == 1:
                classes = list()
                if spn_idx_right in self.col_edges:
                    classes.append('col_edge')
                for key in self.CLASS_VALUES_COL:
                    try:
                        if self.df.semantics.col[spn_idx_left] == key:
                            classes.append(self.CLASS_VALUES_COL[key])
                    except TypeError:
                        pass
                classes = ' '.join(classes)
                classes = f' {classes}' if classes != '' else ''

            html_repr = (
                f'<th class="{prefix}_name{classes}"'
                f'{spn_str}>{idx_name}</th>\n'
                )
            idx_names.append(html_repr)

            if axis == 0:
                nones = [None] * (span[1] - 1)
                idx_names.extend(nones)

            spn_idx_left += span[1]
        return idx_names

    def find_edges(self, index, nlevels, edge_level=-1):
        """
        Return the index edges based on the edge level.
        """

        idx_edges = list()
        if nlevels == 1:
            if edge_level == 1:
                idx_names = index
            else:
                idx_names = list()
        else:
            idx_names = [key[: edge_level] for key in index]
        spans = self.find_spans(idx_names)
        spans = [span[1] for span in spans]
        idx_edges = list(itertools.accumulate(spans))
        idx_edges = [idx_edge - 1 for idx_edge in idx_edges]
        return idx_edges

    @staticmethod
    def find_spans(idx_vals):
        """
        Return values and their (row/column) spans from an index.
        """

        spans = list()
        for val in idx_vals:
            try:
                if not val == spans[-1][0]:
                    spans.append((val, 1))
                else:
                    val_tup = spans.pop(-1)
                    new_val_tup = val, val_tup[1] + 1
                    spans.append(new_val_tup)
            except IndexError:
                spans.append((val, 1))
        return spans


class Style:
    path_to_css = Path(__file__).resolve().parent
    DEFAULT = 'kindofblue'

    def __init__(self, style=None):
        if style is not None:
            self._check_style(style)
        self.style = self.DEFAULT if style is None else style

    def _check_style(self, style):
        if not isinstance(style, str):
            raise TypeError('Input has to be the name (string) of a style.')
        if style not in self.styles:
            raise ValueError(
                f'Style \'{style}\' not found, '
                f'available styles are: {self.styles}'
                )

    @property
    def styles(self):
        return [
            path.stem.split('_')[1]
            for path in self.path_to_css.glob('table_*.css')
            ]

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, value):
    """
        Setter for the style.
        Also loads the edge levels in a style if any are defined.
    """

        self._style = value

        edge_lvls = {'row': None, 'col': None}
        axes = ['row', 'col']
        lines = self.css.split('\n', 4)[1:3]
        for line in lines:
            try:
                elems = line.split('=')
                for axis in axes:
                    if axis in elems[0]:
                        edge_lvls[axis] = elems[1].strip()
            except IndexError:
                continue
        self.edge_lvl_row = edge_lvls['row']
        self.edge_lvl_col = edge_lvls['col']

    @property
    def css(self):
        style_path = self.path_to_css / f'table_{self.style}.css'
        return style_path.read_text()

    if not transpose:
        df = df.copy()
        df.semantics.col = col_semantics
        df.semantics.row = row_semantics
    else:
        df = df.T.copy()
        df.semantics.col = row_semantics
        df.semantics.row = col_semantics
    return df


def ct(
    df,
    row_fields,
    column_fields,
    ignore_nan=False,
    totals_name='Totals',
    totals_col=True,
    totals_row=True,
    perc_cols=False,
    perc_axis='grand',
    perc_round=1,
    name_abs='abs',
    name_rel='%',
):
    """
    Create frequency crosstab for selected categories mapped to specified row and column fields. Group by and count selected categories in df. Then set to rows and columns in crosstab output.

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
    :param totals_col: boolean, default True
        Add totals column.
    :param totals_row: boolean, default True
        Add totals row.
    :param perc_cols: boolean, default False
        Add relative frequency per column
    :param perc_axis: {'grand', 'index', 'columns'}, or {0,1}, default 'grand'
        'grand' - Calculate percentages from grand total.
        'index', 0 - Calculate percentages from row totals.
        'columns', 1 - Calculate percentages from column totals.
    :param round: int or None, default 1
        Number of decimal places to round percentages to. If None percentages will not be rounded.
    :param name_abs: str, default 'abs'
        Name for absolute column.
    :param name_rel: str, default '%'
        Name for relative column.

    Returns
    =======
    :crosstab: DataFrame
    """

    df = df.copy()

    if not column_fields:
        column_fields = '_tmp'
        df[column_fields] = '_tmp'

    # assure row and column fields are lists
    if not isinstance(row_fields, list):
        row_fields = [row_fields]
    if not isinstance(column_fields, list):
        column_fields = [column_fields]

    margins = totals_col or totals_row

    # set columns to use/select from df
    group_cols = column_fields.copy()
    group_cols.extend(row_fields)

    # fill nan if ignore_nan is False
    if not ignore_nan:
        for col in group_cols:
            if df[col].isnull().values.any():
                if df[col].dtype.name == 'category':
                    df[col] = df[col].cat.add_categories([''])
                df[col] = df[col].fillna('')

    # find column for counting that is not in group_cols
    check = False
    i = 0
    while not check:
        try:
            col = df.columns[i]
            if not col in group_cols:
                check = True
            i += 1
        except:
            df['_tmp'] = '_tmp'
            col = '_tmp'
            check = True

    # pivot table
    df = df.groupby(group_cols)[[col]].count()
    df = df.dropna()
    df = pd.pivot_table(
        df.reset_index(),
        index=row_fields,
        columns=column_fields,
        aggfunc='sum',
        dropna=False,
        margins=margins,
        margins_name=totals_name,
    )
    df = df.dropna(how='all')

    if margins:
        if not totals_col:
            df = df.drop(totals_name, axis=1, level=1)
        if not totals_row:
            try:
                df = df.drop(totals_name, axis=0, level=0)
            except:
                df = df.drop(totals_name, axis=0)
    df.columns = df.columns.droplevel(0)
    df = df.fillna(0)

    # remove row/columns where all values are 0
    df = df.loc[(df != 0).any(axis=1)]
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.astype('int64')

    # try to remove temp column
    try:
        df = df.drop('_tmp', axis=1)
        df.columns.name = ''
    except:
        pass

    # add semantics
    df = add_semantics(df)
    if totals_col:
        df.semantics.col[-1] = 'T'
    if totals_row:
        df.semantics.row[-1] = 'T'

    # percentage columns
    if perc_cols:
        df = add_perc_cols(
            df,
            axis=perc_axis,
            totals='auto',
            round=perc_round,
            name_abs=name_abs,
            name_rel=name_rel,
        )
    return df


def add_perc_cols(
    df,
    axis='grand',
    totals='auto',
    name_abs='abs',
    name_rel='%',
    round=1,
):
    """
    Add percentage columns for all columns in the DataFrame.

    Parameters
    ==========
    :param df: DataFrame

    Optional keyword arguments
    ==========================
    :param axis: {'grand', 'index', 'columns'}, or {0,1}, default 'grand'
        'grand' - Calculate percentages from grand total.
        'index', 0 - Calculate percentages from row totals.
        'columns', 1 - Calculate percentages from column totals.
    :param totals_row: boolean, {'auto'}, default 'auto'
        'auto' - Check automatically (may backfire).
        True - Take the totals from the DataFrame (last row/column/value).
        False - Calculate the totals.
    :param name_abs: string, default 'abs'
        Name of absolute column.
    :param name_rel: string, default '%'
        Name of relative column.
    :param round: int or None, default 1
        Number of decimal places to round percentages to. If None percentages will not be rounded.

    Returns
    =======
    :add_perc_cols: DataFrame
    """

    nrows, ncols = df.shape

    def check_for_totals_col(df, col, totals_mode):
        try:
            if df.semantics.row[-1] == 'T':
                return df.iloc[-1][col]
            else:
                return df[col].sum()
        except:
            total = df.iloc[-1][col]
            if totals_mode == 'auto':
                if not total == df.iloc[:nrows - 1][col].sum():
                    total = df[col].sum()
                return total
            if totals_mode:
                return total
            else:
                return df[col].sum()

    def check_for_totals_row(df, totals_mode):
        try:
            if df.semantics.col[-1] == 'T':
                return df.iloc[:, -1]
            else:
                return df.sum(axis=1)
        except:
            total = df.iloc[:, -1]
            if totals_mode == 'auto':
                if not total.equals(df.iloc[:, :ncols - 1].sum(axis=1)):
                    total = df.sum(axis=1)
                return total
            if totals_mode:
                return total
            else:
                return df.sum(axis=1)

    def check_for_grand_total(df, totals_mode):
        try:
            total_col = df.semantics.col[-1] == 'T'
            total_row = df.semantics.row[-1] == 'T'
            if total_col and total_row:
                return df.iloc[-1, -1]
            elif total_col:
                return df.iloc[:, -1].sum()
            elif total_row:
                return df.iloc[-1, :].sum()
            else:
                return df.sum().sum()
        except:
            total = df.iloc[-1, -1]
            if totals_mode == 'auto':
                if not total == df.iloc[:nrows - 1, :ncols - 1].values.sum():
                    total = check_for_totals_row(df, 'auto').values.sum()
                return total
            elif totals_mode:
                return total
            else:
                return check_for_totals_row(df, False).values.sum()

    def set_total(df, col, axis, totals):
        maparg = {0: check_for_totals_row,
                  'index': check_for_totals_row,
                  1: check_for_totals_col,
                  'columns': check_for_totals_col,
                  'grand': check_for_grand_total,
                  }
        if not axis in [1, 'columns']:
            total = maparg[axis](df, totals)
        else:
            total = maparg[axis](df, col, totals)
        return total

    df_output = copy_df(df)
    df_output = add_semantics(df_output)
    col_semantics = df_output.semantics.col.copy()
    row_semantics = df_output.semantics.row.copy()

    # add column index for labelling percentage columns
    nlevels = df_output.columns.nlevels + 1
    levels = list(range(nlevels))
    levels.append(levels.pop(0))
    df_output = pd.concat([df_output], axis=1, keys=[name_abs]).reorder_levels(
        levels, axis=1)

    # add percentage columns
    for col in df.columns:
        new_col = col, name_rel
        abs_col = col, name_abs
        if isinstance(col, tuple):
            new_col = *col, name_rel
            abs_col = *col, name_abs

        total = set_total(df, col, axis, totals)
def add_perc_cols2(
    df,
    axis='grand',
    totals='auto',
    name_abs='abs',
    name_rel='%',
    round=1,
):
    """
    TBD
    """

    df_out = copy_df(df)
    df_out = add_semantics(df_out)
    col_semantics = df_out.semantics.col.copy()
    row_semantics = df_out.semantics.row.copy()

    # add column index for labelling percentage columns
    nlevels = df_out.columns.nlevels + 1
    levels = list(range(nlevels))
    levels.append(levels.pop(0))
    df_out = pd.concat([df_out], axis=1, keys=[name_abs]).reorder_levels(
        levels, axis=1)

    # add percentage columns
    for col in df.columns:
        new_col = col, name_rel
        abs_col = col, name_abs
        if isinstance(col, tuple):
            new_col = *col, name_rel
            abs_col = *col, name_abs

        total = _find_total(df, col, axis, totals)
        col_idx = df_out.columns.get_loc(abs_col)
        new_cols = df_out.columns.insert(col_idx + 1, new_col)
        col_semantics.insert(col_idx + 1, 'p')
        if col_semantics[col_idx] == 'T':
            col_semantics[col_idx + 1] = 'P'
        df_out = pd.DataFrame(df_out, columns=new_cols)
        df_out[new_col] = df_out[abs_col] / total * 100
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
    except:
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
    except:
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
    except:
        total = df.iloc[-1, -1]
        if totals == 'auto':
            if not total == df.iloc[:nrows - 1, :ncols - 1].values.sum():
                total = _totals_row(df, 'auto').values.sum()
            return total
        elif totals:
            return total
        else:
            return _totals_row(df, False).values.sum()


def add_sub_agg(df, level, axis=0, aggs='sum', labels=None, rounding=1):
    """
    Aggregate within the specified level of a multiindex.
    (sum, count, mean, std, var, min, max)

    Parameters
    ==========
    :param df: DataFrame
    :param level: int
        Level of the multiindex to be used
        for selecting the columns that will be subtotalled.

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
    :param round: int or None, default 1
        Number of decimal places to round aggregation to.
        If None aggregation will not be rounded.

    Returns
    =======
    :sub_agg: DataFrame
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
    if isinstance(rounding, list):
        if not len(aggs) == len(rounding):
            raise IndexError(
                'Not every aggregation has a rounding decimal specified. '
                'Set list element to None if no rounding should be applied.'
                )
    else:
        rounding = [rounding] * len(aggs)

    aggs.reverse()
    labels.reverse()
    rounding.reverse()

    for agg, label, round in zip(aggs, labels, rounding):
        df = _add_sub_agg(
            df, level, axis=axis, agg=agg, label=label, round=round
            )
    return df


def _add_sub_agg(df, level, axis=0, agg='sum', label=None, round=1):
    """
    Main logic to the add_sub_agg function.
    """

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

    # set levels
    nlevels = df.columns.nlevels
    if nlevels < 2:
        raise Exception(
            f'The index is not a multiindex. No subaggregation can occur.')
    if level >= nlevels - 1:
        raise Exception(
            f'The index has {nlevels - 1} useable levels: '
            f'{list(range(nlevels - 1))}. Level {level} is out of bounds.'
            )
    nlevels += 1
    level += 1

    # deal with categorical indexes
    if df_out.columns.levels[level].dtype.name == 'category':
        new_level = df_out.columns.levels[level].add_categories(label)
        df_out.columns.set_levels(new_level, level=level, inplace=True)

    i = level + 1
    while i < (nlevels - 1):
        try:
            new_level = df_out.columns.levels[i].add_categories('')
            df_out.columns.set_levels(new_level, level=i, inplace=True)
        except:
            pass
        i += 1

    # collect column keys for specified level
    content = ['v', 'p']
    v_cols = [elem == 'v' for elem in col_semantics]
    c_cols = [elem in content for elem in col_semantics]

    col_keys = list()
    for col in df.loc[:, v_cols].columns.remove_unused_levels():
        fnd_col = col[: level]
        col_keys.append(fnd_col)
    col_keys = list(dict.fromkeys(col_keys))

    # select groups from table, sum them and add to df
    level_list = list(range(level))
    for key in col_keys:
        # find last key in group
        tbl_grp = (df.loc[:, c_cols]
                     .xs([*key], axis=1, level=level_list, drop_level=False))

        key_last_col = tbl_grp.iloc[:, -1].name
        lst_last_col = list(key_last_col)
        lst_last_col[level] = label

        i = level + 1
        while i < (nlevels - 1):
            lst_last_col[i] = ''
            i += 1
        key_new_col = tuple(lst_last_col)

        # insert new column
        idx_col = df_out.columns.get_loc(key_last_col) + 1
        extended_cols = df_out.insert(idx_col, key_new_col, 0)
        col_semantics.insert(idx_col, semantic_code)
        df_out = pd.DataFrame(df_out, columns=extended_cols)

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
        df_out.update(df_col)

    if axis == 0:
        df_out = df_out.T
        df_out = df_out.astype(original_dtypes)
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
