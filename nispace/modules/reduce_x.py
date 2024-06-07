import numpy as np
from sklearn.decomposition import PCA, FastICA
try:
    from factor_analyzer import FactorAnalyzer
    _FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    _FACTOR_ANALYZER_AVAILABLE = False

from .. import lgr
from ..stats.coloc import corr


def _reduce_dimensions(data, method="pca", n_components=None, min_ev=None, 
                       fa_method="minres", fa_rotation="promax",
                       seed=None):
    
    # to array
    data = np.array(data)
    
    # set n_components to max number if min explained variance is given
    n_components = data.shape[1] if (n_components is None) | (min_ev is not None) else n_components
    lgr.info(f"Performing dimensionality reduction using {method} (max components: "
             f"{n_components}, min EV: {min_ev}).")
    
    # case pca
    if method=="pca":
        # run pca with all components
        pcs = PCA(n_components=n_components).fit_transform(data)
        ev = np.var(pcs, axis=0) / np.sum(np.var(data, axis=0))
        # find number of components that sum up to total EV of >= min_ev
        if min_ev is not None:
            total_ev = 0
            for i, e in enumerate(ev):
                total_ev += e
                if total_ev>=min_ev:
                    n_components = i+1
                    lgr.info(f"{n_components} PC(s) explain(s) a total variance of "
                             f"{np.sum(ev[:n_components]):.04f} >= {min_ev} ({ev[:n_components]}).")
                    break
        # cut components & ev
        components = pcs[:,:n_components]
        ev = ev[:n_components]
        lgr.info(f"Returning {n_components} principal component(s).")
    
    # case ica
    elif method=="ica":
        components = FastICA(n_components=n_components, random_state=seed, max_iter=1000)\
            .fit_transform(data) 
        ev = None
        lgr.info(f"Returning {n_components} independent component(s).")
 
    # case fa
    elif method=="fa":
        # handle FactorAnalyzer as optional dependency
        if not _FACTOR_ANALYZER_AVAILABLE:
            lgr.critical_raise("Optional dependency: FactorAnalyzer. Run 'pip install "
                               "factor-analyzer' in your environment to use factor analysis.",
                               ImportError)
        else:
            from factor_analyzer import FactorAnalyzer
        
        # find number of components without rotation that sum up to total EV of >= min_ev
        if min_ev is not None:
            fa = FactorAnalyzer(n_factors=n_components, method=fa_method, rotation=None)
            fa.fit(data)
            ev = fa.get_factor_variance()[2]
            if ev[-1]<min_ev:
                n_components -= 1
                lgr.warning(f"Given min EV ({min_ev}) > max possible EV ({ev[-1]:.02f})! "
                            f"Using max factor number ({n_components}).") 
            else:
                n_components = [i for i in range(len(ev)) if (ev[i] > min_ev)][1]
                lgr.info(f"{n_components} factor(s) explain(s) a total variance of "
                         f"{ev[n_components]:.02f} >= {min_ev}.")
        # run actual factor analysis
        fa = FactorAnalyzer(n_factors=n_components, method=fa_method, rotation=fa_rotation)
        fa.fit(data)
        components = fa.transform(data)
        loadings = fa.loadings_
        ev = fa.get_factor_variance()[1]
        lgr.info(f"Returning {n_components} factor(s).")
        
    else:
        lgr.critical_raise(f"method = '{method}' not defined!", ValueError)
    
    # get PCA and ICA "loadings"
    if method in ["pca", "ica"]:
        loadings = np.zeros((data.shape[1], n_components))
        for c in range(n_components):
            for r in range(data.shape[1]):
                loadings[r, c] = corr(x=data[:, r], y=components[:, c], rank=False)
    ## return
    return components, ev, loadings