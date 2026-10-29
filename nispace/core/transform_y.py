import re
import warnings
import numpy as np
import pandas as pd

from ..utils.utils import _lower_strip_ws
from ..stats.effectsize import (cohen_nan_fast, cohen_paired_nan_fast, hedges_nan_fast,
                                zscore_nan_fast, rzscore_nan_fast,
                                prc_fast, logfc_fast, centile_fast)


def _dummy_code_groups(groups):
    # must run before sorted(set(groups)) below -- a mix of None/nan with a
    # non-numeric type (e.g. strings) makes sorted() raise an unrelated
    # TypeError before this check would otherwise be reached
    if pd.Series(groups).isnull().any():
        raise ValueError("Input contains nan's!")

    unique_elements = _group_levels(groups)

    if len(unique_elements) > 2:
        raise ValueError("Input contains more than two unique elements.")

    # If there's only one unique element, map it to 0.
    if len(unique_elements) == 1:
        return [0] * len(groups)

    # Map the first (smallest or alphabetically first) unique element to 0 and the second to 1.
    mapping = {unique_elements[0]: 0, unique_elements[1]: 1}
    return np.array([mapping[item] for item in groups])


def _group_levels(groups):
    """Sorted unique group labels; `_dummy_code_groups` codes them 0 (= a), 1 (= b)."""
    return sorted(list(set(groups)))


def _label_tokens(label):
    """Strings by which a group label can be written in a formula (lower-case, no spaces)."""
    tokens = {_lower_strip_ws(str(label))}
    # 1.0 can be written as "1"
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        tokens.add(str(int(label)))
    return tokens


def _resolve_group_labels(formula, levels, return_used=False):
    """Replace group labels used as formula arguments by the placeholders a/b.

    Lets users write the direction explicitly, e.g. ``"hedges(AN,HC)"`` instead of
    ``"hedges(a,b)"``, whose meaning depends on the sort order of the labels.

    Parameters
    ----------
    formula : str
        Lower-case formula without whitespace (as from `_lower_strip_ws`).
    levels : list
        Sorted group labels (`_group_levels`): ``levels[0]`` -> a, ``levels[1]`` -> b.
    return_used : bool, default False
        Also return whether the formula contains group labels.

    Returns
    -------
    str
        The formula with placeholders only. Unchanged if it contains no labels.
    bool
        Only if ``return_used``: True if any argument was a group label.
    """
    used = []
    lookup = {}
    for placeholder, label in zip("ab", levels):
        for token in _label_tokens(label):
            lookup.setdefault(token, set()).add(placeholder)

    # arguments are delimited by "(", ")", ",", "-" or the string ends; labels come first in
    # the alternation so that labels containing "-" (e.g. "ses-1") are matched as a whole
    alts = "|".join(re.escape(t) for t in sorted(lookup, key=len, reverse=True))
    pattern = re.compile(rf"(?<![^(,\-])({alts + '|' if alts else ''}[^(),\-]+)(?![^),\-])")

    def _sub(match):
        token = match.group(1)
        if token in lookup:
            if len(lookup[token]) > 1:
                raise ValueError(f"Group labels {levels} are identical after lower-casing and "
                                 f"removing whitespace and cannot be used in the formula "
                                 f"'{formula}'. Use 'a' and 'b' or rename the groups.")
            placeholder = next(iter(lookup[token]))
            if token in ("a", "b", "y") and token != placeholder:
                raise ValueError(f"Group label '{token}' in formula '{formula}' is also a formula "
                                 f"placeholder (a, b, y) with a different meaning: '{token}' is "
                                 f"group {placeholder}. Rename the groups.")
            used.append(token)
            return placeholder
        if token in ("a", "b", "y"):
            return token
        raise ValueError(f"'{token}' in formula '{formula}' is neither a group label "
                         f"({', '.join(map(str, levels))}) nor a placeholder (a, b, y).")

    formula_ab = pattern.sub(_sub, formula)
    return (formula_ab, len(used) > 0) if return_used else formula_ab


# formulas that return one map per subject of the first argument, relative to the second
_RELATIVE_FORMULAS = {"zscore", "rzscore", "centile"}


def _describe_transform(formula, levels):
    """Plain-language direction of a two-group formula, e.g. "positive where 'AN'>'HC'".

    ``formula`` uses placeholders only (after `_resolve_group_labels`). Returns None for
    formulas that do not involve groups a/b.
    """
    formula, formula_wildcard = _normalize_formula(formula)
    name = formula_wildcard.split("(")[0]
    args = [arg for arg in _args_to_tuple(formula) if arg is not None]
    if not set(args) & {"a", "b"} or len(levels) < 2:
        return None
    lab = {"a": f"'{levels[0]}'", "b": f"'{levels[1]}'"}
    if len(args) == 1:
        return f"uses group {lab[args[0]]} only"
    first, second = lab[args[0]], lab[args[1]]
    if name in _RELATIVE_FORMULAS:
        return f"one map per {first} subject, relative to the {second} distribution"
    return f"positive where {first}>{second}"


def _num_code_subjects(subjects):
    # must run before np.unique() below -- a mix of None with a non-numeric
    # type (e.g. strings) makes np.unique's internal sort raise an unrelated
    # TypeError before this check would otherwise be reached
    if pd.Series(subjects).isnull().any():
        raise ValueError("Input contains nan's!")

    unique_subjects, unique_counts = np.unique(subjects, return_counts=True)

    if (unique_counts > 2).any():
        raise ValueError(f"Input contains more than two unique elements at position(s) {np.where(unique_counts > 2)[0]}.")
    
    if (unique_counts < 2).any():
        raise ValueError(f"Input contains less than two unique elements at position(s) {np.where(unique_counts < 2)[0]}.")
    
    # Map the first (smallest or alphabetically first) unique element to 0 and the second to 1.
    mapping = {unique_subjects[i]: i for i in range(len(unique_subjects))}
    return np.array([mapping[item] for item in subjects])             

   
def return_arr(x):
    """Identity function; backs the ``"*"`` (no transform) entry of `_FUN_MAP`."""
    return x

def mean0(x):
    """NaN-aware mean along axis 0; backs the ``"mean(*)"`` `_FUN_MAP` formula."""
    return np.nanmean(x, axis=0)

def median0(x):
    """NaN-aware median along axis 0; backs the ``"median(*)"`` `_FUN_MAP` formula."""
    return np.nanmedian(x, axis=0)

def std0(x):
    """NaN-aware sample SD (ddof=1) along axis 0; backs the ``"std(*)"`` `_FUN_MAP` formula."""
    return np.nanstd(x, axis=0, ddof=1)

def var0(x):
    """NaN-aware sample variance (ddof=1) along axis 0; backs the ``"var(*)"`` `_FUN_MAP` formula."""
    return np.nanvar(x, axis=0, ddof=1)

def elem_diff(a, b):
    """Elementwise difference ``a - b``; backs the ``"elemdiff(*,*)"`` `_FUN_MAP` formula."""
    return a - b

def mean0_diff(a, b):
    """Difference of `mean0` reductions; backs the ``"meandiff(*,*)"`` `_FUN_MAP` formula."""
    return mean0(a) - mean0(b)

def center0(a, b=None):
    """Mean-center `a` (by its own mean, or `b`'s if given); backs ``"center(*,*)"`` in `_FUN_MAP`."""
    if b is None:
        return a - mean0(a)
    else:
        return a - mean0(b)

def _normalize_formula(formula):
    
    formula = _lower_strip_ws(formula)
    # "-" between the two arguments of a function: "hedges(a-b)" = "hedges(a,b)"
    formula = re.sub(r"^(\w+)\(([aby])-([aby])\)$", r"\1(\2,\3)", formula)
    formula_wildcard = re.sub(r"\b[yab]\b", "*", formula)
    
    if formula_wildcard in ["*-*", "mean(*)-mean(*)", "*-mean(*)"]:
        tmp = formula.replace("mean(", "").replace(")","").replace("-", ",")
        if formula_wildcard == "*-*":
            form = "elemdiff"
        elif formula_wildcard == "mean(*)-mean(*)":
            form = "meandiff"
        elif formula_wildcard == "*-mean(*)":
            form = "center"
        formula = f"{form}({tmp})"
        formula_wildcard = f"{form}(*,*)"
    
    return formula, formula_wildcard
    

def _args_to_tuple(expression):
    # Pattern to match function calls with two arguments, e.g., "cohen(a,b)"
    func_two_args = re.compile(r'(\w+)\((a|b|y),(a|b|y)\)')
    match = func_two_args.match(expression)
    if match:
        # If the pattern matches, return the two arguments as a tuple
        return (match.group(2), match.group(3))
    
    # Pattern for single arguments, e.g., "a", "b", "y", or function calls like "mean(a)"
    single_arg = re.compile(r'(?:\w+\()?(a|b|y)\)?')
    match = single_arg.match(expression)
    if match:
        # If the pattern matches, return the argument and None as a tuple
        return (match.group(1), None)
    
    # If no pattern matches, return a tuple indicating an unrecognized expression
    return ("unrecognized", None)


# Mapping of formula strings to function calls
_FUN_MAP = {
    "*": return_arr,
    "mean(*)": mean0,
    "median(*)": median0,
    "std(*)": std0,
    "var(*)": var0,
    "elemdiff(*,*)": elem_diff,
    "meandiff(*,*)": mean0_diff,
    "center(*,*)": center0,
    "cohen(*,*)": cohen_nan_fast,
    "pairedcohen(*,*)": cohen_paired_nan_fast,
    "hedges(*,*)": hedges_nan_fast,
    "zscore(*)": zscore_nan_fast,
    "zscore(*,*)": zscore_nan_fast,
    "rzscore(*)": rzscore_nan_fast,
    "rzscore(*,*)": rzscore_nan_fast,
    "prc(*,*)": prc_fast,
    "logfc(*,*)": logfc_fast,
    "centile(*)": centile_fast,
    "centile(*,*)": centile_fast,
}

_PAIRED_FORMULAS = {"elemdiff(*,*)", "pairedcohen(*,*)", "prc(*,*)"}


def _parse_transform_formula(formula):
    """formula -> (formula, formula_wildcard, trans_fun, args, paired). Single source
    of parsing/validation shared by _get_transform_fun and other callers that need
    the parsed formula without the full apply_transform closure."""
    formula, formula_wildcard = _normalize_formula(formula)
    if formula_wildcard not in _FUN_MAP:
        raise ValueError(f"Provided formula ('{formula_wildcard}'; * = a|b|y) not allowed! "
                         f"Must be one of: {list(_FUN_MAP.keys())}.")
    paired = formula_wildcard in _PAIRED_FORMULAS
    trans_fun = _FUN_MAP[formula_wildcard]
    args = _args_to_tuple(formula)
    args = [arg for arg in args if arg is not None]
    return formula, formula_wildcard, trans_fun, args, paired


def _get_transform_fun(formula, return_df=True, return_paired=False,
                       dtype=np.float32, ignore_nan_warnings=False):
    # parse and validate the formula
    formula, formula_wildcard, trans_fun, args, paired = _parse_transform_formula(formula)

    # create transform function
    def apply_transform(y=None, groups=None, subjects=None):
        if y is None:
            raise ValueError("y must not be None!")
        
        arrays = {"y": np.array(y, dtype=dtype)}
        
        if groups is not None:
            arrays["a"] = arrays["y"][groups==0, :]
            arrays["b"] = arrays["y"][groups==1, :]
            if subjects is not None:
                arrays_order = {
                    "a": np.argsort(subjects[groups==0]),
                    "b": np.argsort(subjects[groups==1])
                }
                arrays["a"] = arrays["a"][arrays_order["a"]]
                arrays["b"] = arrays["b"][arrays_order["b"]]
        
        if ignore_nan_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
                warnings.filterwarnings("ignore", "Degrees of freedom <= 0 for slice", RuntimeWarning)
                res = trans_fun(*[arrays[arg] for arg in args]).astype(dtype)
        else:
            res = trans_fun(*[arrays[arg] for arg in args]).astype(dtype)
  
        # ensure orientation and 2-dimensionality of output
        res = np.atleast_2d(res)
        
        # get correct indices if output as df
        if return_df:
            cols = y.columns
            if res.shape[0]==1:
                idc = [formula_wildcard.split("(")[0]]
            elif args[0] == "y":
                idc = y.index
            elif args[0] in ["a", "b"]:
                idc = y.index[groups==0 if args[0]=="a" else groups==1]
                if subjects is not None:
                    idc = idc[arrays_order[args[0]]]
            res = pd.DataFrame(res, columns=cols, index=idc, dtype=dtype)
            
        # return
        return res
    
    return apply_transform if return_paired == False else (apply_transform, paired) 


# def _get_transform_interpreter(formula, return_df=True):

#     ## Prepare the interpreter
#     ae = Interpreter()
    
#     ## prepare formula
#     if isinstance(formula, str):
#         # strip formula of whitespaces and make lower-case
#         formula = formula.lower().replace(" ","")
#         # test for allowed options
#         formula_options_return1d = [
#             "mean(*)", "median(*)", "std(*)", "var(*)",
#             "mean(*)-mean(*)", "md",
#             "cohen(*,*)", "pairedcohen(*,*)", "hedges(*,*)", "pairedhedges(*,*)", 
#         ]
#         formula_options_return2d = [
#             "*",
#             "*-*",
#             "*-mean(*)", "*-median(*)",
#             "*/std(*)", "*/var(*)",
#             "prc(*,*)",
#             "(*-mean(*))/std(*)", "(*-median(*))/std(*)",
#             "zscore(*,*)", "zscore(*)"
#         ]
#         formula_options = formula_options_return1d + formula_options_return2d
#         formula_wildcard = re.sub(r"\b[yab]\b", "*", formula)
#         if formula_wildcard not in formula_options:
#             raise ValueError(f"Provided formula ('{formula_wildcard}'; * = y, a, or b) not allowed! "
#                              f"Must be one of: {formula_options}")
            
#     else:
#         raise TypeError(f"Input formula must be of type string, not {type(formula)}!")
    
#     ## General functions
#     ae.symtable["mean"] = lambda x:np.mean(x, axis=0)
#     ae.symtable["median"] = lambda x:np.median(x, axis=0)
#     ae.symtable["std"] = lambda x:np.std(x, axis=0)
#     ae.symtable["var"] = lambda x:np.var(x, axis=0)
#     ae.symtable["md"] = lambda a,b:np.mean(a, axis=0) - np.mean(b, axis=0)
#     ae.symtable["cohen"] = cohen
#     ae.symtable["pairedcohen"] = cohen_paired
#     ae.symtable["hedges"] = hedges
#     ae.symtable["pairedhedges"] = hedges_paired
#     ae.symtable["zscore"] = zscore
#     ae.symtable["prc"] = prc

#     ## function to apply transform to y or y-by-group data
#     def apply_transform(y=None, groups=None):
        
#         # prepare y data
#         if y is not None:
#             ae.symtable["y"] = np.array(y)
            
#         # prepare group data
#         if (y is not None) & (groups is not None):
#             if not return_df:
#                 a = np.array(y)[groups==0, :]
#                 b = np.array(y)[groups==1, :]
#             else:
#                 a = y.loc[groups==0, :]
#                 b = y.loc[groups==1, :]
#             ae.symtable["a"] = np.array(a)
#             ae.symtable["b"] = np.array(b)
            
#         # apply
#         try:
#             out = ae(formula)
#         except Exception as e:
#             return str(e)
        
#         # ensure orientation and 2-dimensionality of output
#         out = np.atleast_2d(out)
          
#         # get correct indices if output as df
#         if return_df:
#             columns = y.columns
#             if formula_wildcard in formula_options_return1d:
#                 if formula_wildcard=="mean(*)-mean(*)":
#                     indices = ["md"]
#                 else:
#                     indices = [formula_wildcard.split("(")[0]]
#             else:
#                 index_df = formula[0] if formula_wildcard.startswith("*") else formula[formula_wildcard.find("(")+1]
#                 indices = locals()[index_df].index 
#             out = pd.DataFrame(out, columns=columns, index=indices)
            
#         # return
#         return out
        
#     # return function
#     return apply_transform