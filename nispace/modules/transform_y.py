import re
import numpy as np
import pandas as pd

from ..stats.effectsize import *


def _dummy_code_groups(groups):
    unique_elements = sorted(list(set(groups)))
    
    if pd.Series(groups).isnull().any():
        raise ValueError("Input contains nan's!")

    if len(unique_elements) > 2:
        raise ValueError("Input contains more than two unique elements.")

    # If there's only one unique element, map it to 0.
    if len(unique_elements) == 1:
        return [0] * len(groups)

    # Map the first (smallest or alphabetically first) unique element to 0 and the second to 1.
    mapping = {unique_elements[0]: 0, unique_elements[1]: 1}
    return np.array([mapping[item] for item in groups])


def _num_code_subjects(subjects):
    unique_subjects, unique_counts = np.unique(subjects, return_counts=True)
    
    if pd.Series(subjects).isnull().any():
        raise ValueError("Input contains nan's!")
    
    if (unique_counts > 2).any():
        raise ValueError(f"Input contains more than two unique elements at position(s) {np.where(unique_counts > 2)[0]}.")
    
    if (unique_counts < 2).any():
        raise ValueError(f"Input contains less than two unique elements at position(s) {np.where(unique_counts < 2)[0]}.")
    
    # Map the first (smallest or alphabetically first) unique element to 0 and the second to 1.
    mapping = {unique_subjects[i]: i for i in range(len(unique_subjects))}
    return np.array([mapping[item] for item in subjects])             

   
def return_arr(x):
    return x

def mean0(x):
    return np.nanmean(x, axis=0)

def median0(x):
    return np.nanmedian(x, axis=0)

def std0(x):
    return np.nanstd(x, axis=0, ddof=1)

def var0(x):
    return np.nanvar(x, axis=0, ddof=1)

def elem_diff(a, b):
    return a - b

def mean0_diff(a, b):
    return mean0(a) - mean0(b)

def center0(a, b=None):
    if b is None:
        return a - mean0(a)
    else:
        return a - mean0(b)


def _normalize_formula(formula):
    
    formula = formula.lower().replace(" ", "")
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


def _get_transform_fun(formula, return_df=True, return_paired=False, dtype=np.float32):
    # normalize the formula
    formula, formula_wildcard = _normalize_formula(formula)
    
    # Mapping of formula strings to function calls
    fun_map = {
        "*": return_arr,
        "mean(*)": mean0,
        "median(*)": median0,
        "std(*)": std0,
        "var(*)": var0,
        "elemdiff(*,*)": elem_diff,
        "meandiff(*,*)": mean0_diff,
        "center(*,*)": center0,
        "cohen(*,*)": cohen_nan,
        "pairedcohen(*,*)": cohen_paired_nan,
        "hedges(*,*)": hedges_nan,
        "zscore(*)": zscore_nan,
        "zscore(*,*)": zscore_nan,
        "prc(*,*)": prc
    }
    
    # validate the formula
    if formula_wildcard not in fun_map.keys():
         raise ValueError(f"Provided formula ('{formula_wildcard}'; * = a|b|y) not allowed! "
                          f"Must be one of: {list(fun_map.keys())}.")
         
    # paired
    if formula_wildcard in ["elemdiff(*,*)", "pairedcohen(*,*)", "pairedhedges(*,*)", "prc(*,*)"]:
        paired = True
    else:
        paired = False
    
    # transform function
    trans_fun = fun_map[formula_wildcard]
    
    # arguments
    args = _args_to_tuple(formula)
    
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
                
        res = trans_fun(*[arrays[arg] for arg in args if arg is not None]).astype(dtype)
        
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