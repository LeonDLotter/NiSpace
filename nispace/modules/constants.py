
_PARCS = ["schaefer100melbournes1", "schaefer200melbournes2", "schaefer300melbournes3", "hcpex", 
          "desikankilliany", "destrieux"]
_PARCS_NICE = ["Schaefer100MelbourneS1", "Schaefer200MelbourneS2", "Schaefer300MelbourneS3", "HCPex", 
               "DesikanKilliany", "Destrieux"]
_PARCS_DEFAULT = "schaefer200melbournes2"

_DSETS_VERSION = "v0"
_DSETS = ["pet", "mrna", "brainmap"]
_DSETS_NICE = ["PET", "mRNA", "BrainMap"]
_DSETS_MAP = ["pet", "brainmap"]
_DSETS_TAB = ["mrna"]

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



