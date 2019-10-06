"""
This module loads the settings from config.ini.
"""


# standard library
import configparser
from collections import namedtuple
from pathlib import Path


PATH_LIB = Path(__file__).resolve().parent
CFG_FILE = PATH_LIB / 'config.ini'
encoding = 'utf-8'  # encoding of ini file only

QUOTECHAR = '"'
SEPARATOR = ','
BOOLEAN_STATES = {
    'true': True, 'false': False,
    't':    True, 'f':     False,
    '1':    True, '0':     False,
    'yes':  True, 'no':    False,
    'on':   True, 'off':   False,
}


def load_ini(filename):
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(filename, encoding=encoding)
    return ini


def config_from_ini(ini):
    Config = namedtuple('Config', [section for section in ini])
    sections = list()
    for section in ini:
        sections.append(get_section(ini, section))
    return Config._make(sections)


def ntuple_from_section(ini, section):
    return namedtuple(section, [item for item in ini[section]])


def get_section(ini, section, func=None):
    Section = ntuple_from_section(ini, section)
    values = list()
    for item in ini[section]:
        value = parse_value(ini[section][item])
        if func:
            value = func(value)
        values.append(value)
    return Section._make(values)


def parse_value(value):
    value = value.strip('\n ')
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass

    if value[:1] == QUOTECHAR and value[-1:] == QUOTECHAR:
        return value.strip(QUOTECHAR)
    if value[:1] == '[' and value[-1:] == ']':
        return get_list(value[1:-1].strip('\n '))
    if any(item == value.lower() for item in BOOLEAN_STATES):
        return BOOLEAN_STATES[value.lower()]
    return value


def chunker(value):
    def chunk(value, idx):
        return value[:idx], value[idx+1:].strip('\n ')

    if len(value) == 0:
        return None, None
    elif value[0] == QUOTECHAR:
        idx = value.find(value[0], 1) + 1
        return chunk(value, idx)
    else:
        idx = value.find(SEPARATOR)
        if idx == -1:
            return value, None
        return chunk(value, idx)


def get_list(value):
    lst = []
    while True:
        chunk, value = chunker(value)
        if chunk is not None:
            lst.append(parse_value(chunk))
        if value is None:
            break
    if len(lst) > 1:
        return lst
    return lst


# easy access to settings and paths
config = load_ini(CFG_FILE)
TABLES = get_section(config, 'TABLES')
PATHS = get_section(
    config,
    'PATHS',
    func=lambda x: PATH_LIB / x[2:] if x.startswith('./') else Path(x)
)
