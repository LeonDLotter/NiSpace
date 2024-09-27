
_PARCS = ["schaefer100", "schaefer200", "schaefer300", "hcp", "desikankilliany", "destrieux"]
_PARCS_NICE = ["Schaefer100", "Schaefer200", "Schaefer300", "HCP", "DesikanKilliany", "Destrieux"]
_PARCS_DEFAULT = "schaefer200"

_DSETS = ["pet", "mrna", "rsn"]
_DSETS_NICE = ["PET", "mRNA", "RSN"]
_DSETS_TAB_ONLY = ["mrna"]
_DSETS_CX_ONLY = ["rsn"]
_DSETS_SC_ONLY = []

_COLLECT_DEFAULT = {
    "pet": "UniqueTracers",
    "mrna": "CellTypesPsychEncodeTPM",
    "brainmap": "AllDomainSets",
}

_COLOC_METHODS = {
    "pearson": ["rho"],
    "spearman": ["rho"],
    "partialpearson": ["rho"],
    "partialspearman": ["rho"],
    "slr": ["r2"],
    "mlr": ["r2", "beta", "intercept", "individual"],
    "dominance": ["sum", "total", "individual", "relative"],
    "pls": ["r2", "beta"],
    "pcr": ["r2"],
    "lasso": ["r2", "beta", "alpha"],
    "ridge": ["r2", "beta", "alpha"],
    "elasticnet": ["r2", "beta", "alpha", "l1ratio"],
}
_COLOC_METHODS_PERM = {
    "pearson": ["rho"],
    "spearman": ["rho"],
    "partialpearson": ["rho"],
    "partialspearman": ["rho"],
    "slr": ["r2"],
    "mlr": ["r2", "beta", "individual"],
    "dominance": ["sum", "total", "individual", "relative"],
    "pls": ["r2", "beta"],
    "pcr": ["r2"],
    "lasso": ["r2", "beta"],
    "ridge": ["r2", "beta"],
    "elasticnet": ["r2", "beta"],
}
_COLOC_METHODS_1D = {
    "mlr": ["r2", "intercept"],
    "dominance": ["sum"],
    "pls": ["r2"],
    "pcr": ["r2"],
    "lasso": ["r2", "alpha"], 
    "ridge": ["r2", "alpha"],
    "elasticnet": ["r2", "alpha", "l1ratio"],
}
_COLOC_METHODS_DROPOPT = {
    "pearson": ["rho"],
    "spearman": ["rho"],
    "partialpearson": ["rho"],
    "partialspearman": ["rho"],
    "slr": ["r2"],
    "mlr": ["r2", "beta"],
    "dominance": ["sum", "total", "individual", "relative"],
    "pls": ["r2", "beta"],
    "pcr": ["r2"],
    "lasso": ["r2", "beta", "alpha"],
    "ridge": ["r2", "beta", "alpha"],
    "elasticnet": ["r2", "beta", "alpha", "l1ratio"],
}

_P_TAILS = {
    "pearson": {"rho": "two"},
    "spearman": {"rho": "two"},
    "partialpearson": {"rho": "two"},
    "partialspearman": {"rho": "two"},
    "slr": {"r2": "upper"},
    "mlr": {"r2": "upper", "beta": "two", "individual": "upper"},
    "dominance": {"sum": "upper", "total": "upper", "individual": "upper", "relative": "upper"},
    "pls": {"r2": "upper", "beta": "two"},
    "pcr": {"r2": "upper"},
    "lasso": {"r2": "upper", "beta": "two", "alpha": "upper"},
    "ridge": {"r2": "upper", "beta": "two", "alpha": "upper"},
    "elasticnet": {"r2": "upper", "beta": "two", "alpha": "upper", "l1ratio": "upper"},
}



