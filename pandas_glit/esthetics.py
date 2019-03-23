"""
Module for the FancyTable class.
"""

import itertools
import pandas as pd
from pathlib import Path
from pandas_glit import semantics


class FancyTable:
    """
    Class for adding custom styling to DataFrames.
    """
    CLASS_VALUES_COL = {
        'pc': 'perc_col',
        'pr': 'perc_col',
        'pg': 'perc_col',
        'T': 'totals_col',
        'Pc': 'totals_perc_col',
        'Pr': 'totals_perc_col',
        'Pg': 'totals_perc_col',
        'a': 'agg_col',
        'c': 'agg_col',
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
        max_rows=100,
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

        self.max_rows = max_rows

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
        # copy df and limit rows if df exceeds max rows
        df = semantics.copy_df(self.df)
        if len(df) > self.max_rows:
            head = self.max_rows // 2
            len_tail = self.max_rows - head
            tail = len(df) - len_tail
            filt = [x <= head or x >= tail for x in range(len(df))]
            semantics_row = list(itertools.compress(df.semantics.row, filt))
            semantics_row.insert(head, 'v')
            semantics_col = df.semantics.col
            trunc_row = pd.DataFrame(
                columns=df.columns,
                index=['...'],
                ).fillna('...')
            df = df.head(head).append(trunc_row).append(df.tail(len_tail))
            df.semantics.row = semantics_row
            df.semantics.col = semantics_col
        row_elements = list()

        # indices
        i = 0
        while i < self.nlevels_row:
            idx_names = self.get_idx_keys(df.index, i)
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
                    if df.semantics.col[col_idx] == key:
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

        # cast all values as strings
        for col in df.columns:
            try:
                df[col] = df[col].fillna('')
            except ValueError:
                pass
        values = df.astype(str).values
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
                    if df.semantics.row[idx] == key:
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
