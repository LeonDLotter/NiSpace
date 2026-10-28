# NiSpace: guide for AI assistants

This file is for AI assistants (chat assistants, coding agents) that help people use NiSpace,
or that read this repository to answer questions about it. Humans: start with the
[README](README.rst) and the [documentation](https://nispace.readthedocs.io/).

**What is authoritative.** NiSpace is under active development. Parameter names, defaults,
available datasets and collections change between versions; this file deliberately leaves them
out. The docstrings (`help(nispace.NiSpace.colocalize)` etc.) and the installed version are the
ground truth. If this file and a docstring disagree, the docstring wins. Check the version with
`nispace.__version__`.

---

## 1. What NiSpace is

NiSpace (NeuroImaging Spatial Colocalization Environment) is a Python toolbox that tests whether
brain maps share spatial patterns ("colocalization"): for example, whether a map of cortical
thinning in a patient group follows the distribution of a receptor, a cell type, or a
functional network. It covers the full path: fetching reference data, bringing user data and
reference data into one parcellation, estimating colocalizations with uni- and multivariate
methods, testing them against null models, correcting for multiple comparisons, and plotting.

Beyond spatial correlation, NiSpace is also useful on its own as (section 9):
a large, consistently processed multimodal collection of reference brain maps and parcellations,
a brain-plotting tool, an imaging space transformation tool, and a null-map generator.

A good conceptual entry point to the field, including its pitfalls, is:
Royer et al. (2026). Opportunities and pitfalls of data contextualization in neuroimaging.
*Nature Reviews Neuroscience*, 27, 416–434. https://doi.org/10.1038/s41583-026-01038-0

---

## 2. Where the code is

| Purpose | Where |
|---|---|
| One-call analyses ("workflows") | [`nispace/workflows.py`](nispace/workflows.py) |
| Step-by-step analyses: the `NiSpace` class | [`nispace/api.py`](nispace/api.py) |
| Reference data, parcellations, templates, example data (`fetch_*`) | [`nispace/datasets.py`](nispace/datasets.py) |
| Plotting, incl. `brainplot` | [`nispace/plotting.py`](nispace/plotting.py) |
| Diagnostics (which regions drive a result, shared axes, set overlap) | [`nispace/diagnostics.py`](nispace/diagnostics.py) |
| Null maps (spatial surrogates) | [`nispace/nulls.py`](nispace/nulls.py) |
| Space transformations | [`nispace/transform.py`](nispace/transform.py) |
| Parcellating images | [`nispace/parcellate.py`](nispace/parcellate.py) |
| Statistics as plain functions | [`nispace/stats/`](nispace/stats/) |
| Internals (do not call directly) | [`nispace/core/`](nispace/core/) |

The `NiSpace` class is used in this order:
`NiSpace(x=..., y=..., parcellation=...)` → `fit()` → optionally `clean_y()` / `transform_y()` →
`colocalize()` → `permute()` → `correct_p()` → `get_colocalizations()` / `get_p_values()` /
`plot()`. Each workflow function runs this chain and returns the `NiSpace` object, so results
can be inspected and extended with the same methods.

---

## 3. Choose the analysis

First find out what the user has and what they want to know. Then:

| Data and question | Function | What the p-value tests | Example |
|---|---|---|---|
| One or a few maps (group effect map, meta-analysis, single subject) vs. reference maps | `nispace.colocalization` | Spatial null maps: is the association stronger than with maps of similar spatial autocorrelation? | intro02, intro05 |
| Subject-level maps from two groups or two sessions | `nispace.group_colocalization` | Group-label permutation: is the group-difference map more aligned than for random group splits? | intro04 |
| A map vs. *sets* of reference maps (e.g. genes of a cell type, receptors of a system) | `nispace.xsea`, `nispace.group_xsea` | Set enrichment against null maps of Y (or group-label permutation); not against random sets (see below) | intro10 |
| N matched map pairs (e.g. two modalities per subject) | `nispace.paired_colocalization` | Pair-label permutation (SPICE): is within-pair similarity higher than between-pair? | intro13 |
| Per region, across subjects: does X track Y, or a map track a score? | `nispace.correlate_within_region` | Subject-label permutation per region | intro13 |
| Coordinate-based meta-analysis results (NiMARE) | `nispace.nimare_colocalization`, `nispace.nimare_xsea` | Meta-analysis-specific null maps | intro12 |
| Binary maps (clusters, masks) | workflows with `binary_y=True` | as above | intro14 |
| Which regions drive a result? | `nispace.diagnostics.regional_influence`, `regional_contribution`, `local_colocalization` | Descriptive | intro15 |

Use the workflows for standard cases and the `NiSpace` class for anything custom (several
transforms, covariates, mixing null models, multivariate methods).

**Do not use set resampling** (`permute_sets=True` in the XSEA workflows, `permute("sets")` in
the `NiSpace` class) unless the user has a specific hypothesis that pure set resampling answers:
"is this set more aligned with Y than random sets of the same size drawn from the background
maps?" It does not account for the correlations between the maps within a set, the effective
dimensionality of the set, or the spatial autocorrelation of Y, so it does not test whether Y
is aligned with the set beyond chance. These controls are not implemented. For the usual
question, keep the default null (Y permuted or group labels permuted).

---

## 4. Rules for valid analyses

1. **Never report a parametric p-value for a correlation between brain maps.** Neighbouring
   regions are not independent (spatial autocorrelation), so parametric p-values are far too
   small. Always use a null model (`permute()` or a workflow with `n_perm > 0`).
2. **Choose the null model to fit the design and say what it preserves.** With subject-level
   data, group-label permutation uses the real data's structure. Spatial nulls preserve
   smoothness, and depending on the method also e.g. the offset between cortex and subcortex
   (when cortex and subcortex are randomized separately) or a dominant gradient. When the
   conclusion depends on the null, check it with a second null method and report both.
3. **Fix the family of tested reference maps before seeing results, and correct across all of
   them.** Report the correction method and state its guarantee in words: FDR controls the
   expected *proportion* of false positives among significant results; family-wise methods
   control the probability of *any* false positive. Do not present uncorrected results as
   findings.
4. **Mind the permutation resolution.** With B permutations the smallest possible p-value is
   1/(B+1) (about 2/(B+1) two-sided). With many maps, B must be large enough that corrected
   p-values can reach α at all. NiSpace logs when p-values sit at this floor; report them as
   upper bounds (p ≤ …).
5. **Report correlations on the right scale.** NiSpace stores correlations Fisher-z
   transformed by default (check `colocalize()`'s docstring). Apply `numpy.tanh` before quoting
   them as r or ρ, and average correlations only on the z scale.
6. **Match the method to the data.** Rank-based (Spearman) correlation for continuous maps with
   outliers or skew; Pearson with `binary_y=True` for binary maps.
7. **Check the inputs.** User data and reference data must be in compatible spaces and the same
   parcellation. Decide whether zeros are background or real data (e.g. 0 % overlap in a binary
   map is data). Z-scoring each map across regions is right for colocalization but wrong for
   analyses across subjects (`correlate_within_region`).
8. **Do not suppress NiSpace warnings.** They flag real problems (too few permutations, tiny
   sets, standardization on the wrong axis, degenerate nulls). Pass them on to the user.
9. **Record what is needed to reproduce the analysis:** NiSpace version and commit, seed,
   parcellation, reference dataset and collection, colocalization method, null method,
   number of permutations, multiple-comparison method.
10. **Citations are mandatory.** Reference data come from many original studies, mostly under
    attribution licenses: using the data requires citing the original sources.
    - `fetch_reference()` prints a dataset description, the dataset-level citations, and the
      NiSpace version. With per-map metadata loaded (`return_metadata=True`, or
      `nispace.datasets.fetch_map_info()`), it prints a per-map table of sources and licenses.
      A collection of 30 PET maps can come from 40 publications; all of them must be cited.
    - **Never set `print_references=False` (or silence output) without telling the user**
      that citing is required, and without giving them a summary of the sources they need to
      cite (dataset, and per map where applicable). If output is hidden for brevity, say so and
      show where the full list comes from.
    - Also cite the parcellation, NiSpace itself, and the methods used (null-map methods,
      SPICE, etc.); docstrings name the references. See the docs'
      [citation page](https://nispace.readthedocs.io/en/latest/citation.html). BibTeX entries
      for the references cited in the code and documentation are in
      [`docs/references.bib`](docs/references.bib).

---

## 5. Interpretation

**What a colocalization is.** A correlation across regions between two maps: do regions that
are high in one map tend to be high (or low) in the other? Whether it describes a group or an
individual depends on the inputs. A group effect map against group-average reference maps is a
group-level finding. Per-subject maps (e.g. each patient's deviation from controls) give one
colocalization per subject, and subject-specific reference maps (e.g. a subject's own PET scan)
make it fully individual. Most reference maps, however, come from other people and other
studies; they are not the study participants' own biology, and they are not normative
(section 6).

**Specificity is the central question.** A significant colocalization says that the pattern
aligns with this map, not that it aligns *specifically* with this map or with the biology it is
named after. Reference maps are strongly correlated with each other: many follow the same
large-scale axes (gray-matter density, cortex vs. subcortex, anterior-posterior and
superior-inferior gradients, the sensorimotor-association axis), PET tracers bind more than
one target, and mRNA is not protein. Before interpreting a single map, look deeper:

- **How do the reference maps correlate with each other?** `nsp.get_x()` returns the
  parcellated reference maps; correlate them (e.g. `nsp.get_x().T.corr("spearman")`). A hit
  among many correlated maps is a hit for the family, not for one member.
- **Shared spatial axes:** `nispace.diagnostics.shared_axes()` checks whether the user's map and
  the reference maps share known axes. `nispace.datasets.fetch_axes()` provides the axes;
  `NiSpace(z=...)` with a partial method (`colocalize("partialspearman")`) controls for them.
- **Unique vs. shared contributions:** multivariate methods, e.g. `colocalize("dominance")`
  (how much each map adds beyond the others) or `"mlr"`. They help, but cannot fully separate
  intercorrelated maps: shared variance has to be assigned somehow (dominance analysis splits
  it, regression weights become unstable), and a map that adds little may still be the more
  relevant one.
- **Cortex vs. subcortex:** after a whole-brain analysis, always check whether the result holds
  within the cortex alone (re-run with a cortex-only parcellation), and within the subcortex
  alone if the subcortical parcellation has enough regions. Many maps differ strongly in their
  mean level between cortex and subcortex (e.g. most PET maps). With few subcortical regions
  far from the cortical values, this offset alone can produce a strong whole-brain correlation.
  Some null models account for the offset partly or fully, e.g. spatial nulls that
  randomize cortex and subcortex separately, or group-label permutation, which keeps the data's
  structure. A cortex-only analysis is fine when it fits the question.
- **Set overlap and similarity:** `nispace.diagnostics.compare_sets()` tests whether sets
  (e.g. gene sets of two cell types) actually differ in members and in their mean maps. NiSpace
  also warns when maps belong to several sets.
- **Where the effect comes from:** `regional_influence()`, `regional_contribution()` and
  `local_colocalization()` in `nispace.diagnostics`; `plot(kind="correlation")` to look at the
  scatter itself.
- **What each map is:** `nispace.datasets.fetch_map_info()` (tracer, target, sample, source).

**The null models answer different questions.** Spatial nulls: is the alignment stronger than
with random maps of similar smoothness? Group-label permutation: is the group-difference map
more aligned than maps from random group splits? They can disagree, and the disagreement is
informative.

**Other points.**
- The sign depends on the contrast direction (which group is subtracted from which).
- XSEA tests a set as a whole, not each member.
- With few regions or very smooth maps, the effective number of independent observations is
  small: a large correlation can be non-significant, and that is correct.
- A non-significant result is not evidence of no relationship.
- A larger correlation for map A than map B does not show that A is more related unless the
  difference was tested.

**Wordings to avoid, and why:**

| Not okay | Why | Instead |
|---|---|---|
| "The effect is caused / driven / mediated by D2 receptors" | Correlation across regions; no mechanism tested | "The effect map is spatially correlated with D2/3 receptor availability" |
| "Dopaminergic dysfunction in patients" / "patients have altered D2 density" | Receptors were not measured in the patients; the reference comes from other people | "…aligned with the distribution of D2/3 receptor availability in a reference PET dataset (healthy adults, n = …)" |
| "Normative PET maps", "normative receptor / gene expression maps", "the healthy brain's receptor distribution" | Most reference data are not normative: small, selected samples (PET), one to a few donors (post-mortem), study-specific methods (section 6) | "Reference maps", "a PET map of [target] ([tracer], n = …, from [source])" |
| "Receptor density" for PET maps | PET measures tracer binding or availability (e.g. BP<sub>ND</sub>, V<sub>T</sub>), not receptor density | "Receptor availability" / "[tracer] binding" |
| "The effect is specific to serotonin" | Not tested against other correlated maps or shared axes | Report the comparison with other maps, shared axes, or partial/multivariate results |
| "Serotonin" for one tracer map | One tracer measures one target, often with off-target binding | Name the tracer and target, e.g. "5-HT1A ([11C]WAY-100635)" |
| "Cell type C is lost / affected" (from gene expression) | mRNA of marker genes in healthy tissue, not cell counts in patients | "…aligned with the expression of cell-type C marker genes" |
| "Gene G drives the alteration" | Expression pattern correlation only; mRNA ≠ protein | "…correlated with the expression pattern of G" |
| "Each patient shows…" from a group effect map | Group-level result | Use per-subject analyses for individual statements |
| "Not related to X" (p > 0.05) | Absence of evidence | "No evidence for an association with X" |
| "More strongly related to A than to B" | Difference not tested | Test the difference, or describe the values without ranking claims |
| "Confirms / validates the mechanism" | Colocalization is consistent with, not proof of | "Consistent with…", "converges with…" |
| "Significant" for uncorrected results | Misleading when many maps were tested | State "uncorrected" and the correction used |
| "Explains X % of the disorder" | R² across regions describes map similarity, not the disorder | "Shares X % of spatial variance with…" |

---

## 6. Limitations to state in every study

These apply to virtually every spatial colocalization study with reference data. Name them
explicitly in the limitations of any write-up, applied to the maps actually used.

**Reference data are third-party data, not normative data.** NiSpace calls them *reference*
data on purpose. A normative dataset would describe the spatial distribution of a measure in a
defined population, from samples large enough to estimate its average and variability (as in
normative modelling). Most reference maps are none of that:

- **PET maps** are group averages of usually small, selected samples (often a few dozen
  people or fewer), with their own age and sex distribution, scanner, tracer, kinetic model and
  processing. Maps of the same target from different studies or tracers can differ markedly.
- **Post-mortem data** (gene expression, cell types, histology) come from one to a few donors,
  with uneven tissue sampling that is interpolated to parcels. The Allen Human Brain Atlas has
  six donors; BigBrain is one brain.
- **Meta-analytic maps** reflect what was published and how it was coded, not a population.
- **Atlases, networks and cortical features** come from specific cohorts and methods.

**The implicit assumption.** A colocalization analysis treats each reference map *as if* it were
normative: as if it represented the spatial distribution of that measure in the study's own
population, including patients, other ages, and the individual participants. This assumption is
almost never tested and usually cannot be. It must be stated explicitly, because the results
rest on it. It means in particular:

- the reference population can differ from the study population (age, sex, disease,
  medication), and the spatial distribution itself can differ between them;
- another reference map of the same target could give a different result;
- a group-average reference map carries no information about individual differences in the
  measure;
- what the reference map measures is limited by its method: tracers bind more than one target,
  mRNA is not protein, and acquisition, partial-volume correction, spatial normalization and
  parcellation differ between maps.

**What to do in practice.**
- Describe each reference map by what it is (tracer and target, sample size, age, source):
  `nispace.datasets.fetch_map_info()` provides this.
- Where several maps of the same target exist, check whether the result holds across them.
- Use the specificity checks in section 5 (correlations between reference maps, shared axes,
  multivariate models).
- Example limitation sentence: "The PET, gene expression and [...] maps are reference data from
  independent, mostly small samples of healthy adults (post-mortem data: n = 6 donors). Our
  analyses assume that their spatial distributions are representative of the studied
  population, which we could not test."

**Other standing limitations.**
- Colocalization is correlational and across regions: it does not test a mechanism and does
  not show that the measure is altered in the study participants.
- Results depend on the parcellation, the null model, and the set of reference maps tested
  (section 7).

---

## 7. Open research questions: state them as caveats

These are not settled in the field (or in NiSpace); mention them where they affect a result.

- **Spatial null models.** Methods preserve different properties (smoothness, gradients,
  compartment offsets, hemisphere symmetry) and can lead to different conclusions on the same
  data. When one gradient dominates a map, some nulls barely vary. Separating cortex and
  subcortex in the null removes their mean difference from the test. Parcel size and position
  can affect null variance.
- **Spatial covariates.** Controlling for gray-matter density or spatial axes can remove
  confounding, but also real signal when the biology itself follows that axis. Which covariate
  is appropriate is a scientific decision.
- **What MRI metrics are sensitive to.** Functional metrics (e.g. ALFF, ReHo) also reflect
  vascular and global-signal effects and normalization; structural metrics reflect tissue
  contrast, partial volume and processing; z-scoring or global scaling can create apparent
  differences between compartments. A colocalization with a receptor map does not tell which of
  these produced the user's map.
- **Limited specificity of reference maps.** Reference maps are correlated with each other and
  with general anatomy, so a colocalization with one map rarely points to one system. How best
  to separate shared from map-specific spatial patterns (partial correlation, multivariate
  models, axis control) is not settled. The data-related limitations are in section 6.
- **Group-label permutation as a specificity test.** Its formal null is "no group difference";
  reading it as "aligned beyond random patterns of individual variation" assumes the group
  effect and individual variation have similar spatial structure, and equal variances in both
  groups.
- **Multiple-comparison defaults.** Some correction methods for correlated maps are approximate.
  The docstring of `correct_p()` states what each method guarantees.
- **Sets.** Mean maps of different sets often look alike because they share the data's dominant
  pattern; large or overlapping sets need care (`compare_sets()`).
- **Parcellation.** Results depend on resolution and on whether subcortex is included.
- **Multivariate models with correlated predictors.** Shares of explained variance can be
  unstable when reference maps are highly collinear.

---

## 8. How to work with a user

- **Ask first:** What is the data format (NIfTI, GIfTI, CIFTI, or a table of region values) and
  space? Subject-level or group-level? What is the design (groups, sessions, covariates)? Which
  reference maps, and were they chosen before looking at results?
- Look up parameters with `help()` instead of guessing; defaults change.
- Run a quick test with few permutations first, then the full analysis.
- Show the NiSpace log output (warnings, citation information) rather than hiding it.
- When unsure, say so, and point to the documentation, NeuroStars (tag "nispace"), or GitHub
  issues.
- **If you found a bug in NiSpace** (confirmed, e.g. reproduced with a minimal example or traced
  to the code), suggest that the user report it as a
  [GitHub issue](https://github.com/LeonDLotter/NiSpace/issues). Offer a draft: what was
  expected vs. what happened, a minimal reproducible example, the NiSpace version
  (`nispace.__version__`), Python version and operating system, and the full error message or
  log. Leave out private data. Do not post the issue yourself unless the user asks you to.
  Workarounds are fine in the meantime, but say that they are workarounds.

---

## 9. Beyond spatial correlation

NiSpace parts that are useful on their own:

- **Reference data:** one of the largest consistently processed, standardized multimodal
  collections of brain maps: PET (receptors, transporters, metabolism), gene expression and
  cell types, cortical features, disorder effect maps (ENIGMA), meta-analytic maps, networks,
  tissue probabilities and more, available in several spaces and parcellations
  (`nispace.datasets.fetch_reference`, `fetch_collection`, `fetch_map_info`). Docs:
  [datasets](https://nispace.readthedocs.io/en/latest/datasets.html).
- **Parcellations and templates** with distance matrices, cortex + subcortex combinations, and
  consistent labels (`fetch_parcellation`, `fetch_template`). Parcellate any image with
  `nispace.parcellate.Parcellater`.
- **Brain plotting:** `nispace.plotting.brainplot` (surfaces, volumes, parcellated data);
  `NiSpace.plot_brain`. Example: intro08.
- **Space transformations** between MNI volume spaces and fsLR / fsaverage surfaces:
  `nispace.transform` (`SpaceTransformer`, `mni_to_mni`, `surf_to_surf`, `mni_to_surf`).
- **Null maps** for any spatial analysis: `nispace.nulls.generate_null_maps`.
- **Spatial autocorrelation** statistics (Moran's I, variograms, Mantel test):
  `nispace.stats.autocorr`.
- **Spatial axes** (tissue, geometry, cortical hierarchy) as covariates or descriptors:
  `nispace.datasets.fetch_axes`.

Overview with links to the API reference: [Beyond spatial correlation](https://nispace.readthedocs.io/en/latest/beyond.html).

---

## 10. Examples

- [`docs/nb_introduction/`](docs/nb_introduction/): the introduction series, intro01 (concept)
  to intro15. Topics: concept, the NiSpace object, datasets and custom inputs, imaging
  phenotypes and group comparisons, null models, multiple comparisons, visualization, brain
  plotting, workflows, XSEA, advanced topics, NiMARE meta-analyses, paired colocalization
  (SPICE, within-region correlation), binary maps, regional diagnostics.
- [`docs/nb_examples/`](docs/nb_examples/): applied examples (ABIDE cell types, drug challenge,
  ENIGMA, FreeSurfer cross-sectional and longitudinal series). Not available yet: work in
  progress, not part of the rendered documentation; do not rely on them.
- [`docs/nb_benchmarks/`](docs/nb_benchmarks/): evidence for the methods and defaults
  (false-positive rates and power of null models and corrections). More benchmarks will be
  added.
- Rendered versions: https://nispace.readthedocs.io/

## 11. Related resources

- Documentation: https://nispace.readthedocs.io/
- Reference data repository: https://github.com/LeonDLotter/NiSpace-data
- Related tools: [neuromaps](https://netneurolab.github.io/neuromaps/) (maps, transformations,
  nulls), [JuSpace](https://github.com/juryxy/JuSpace) (focused on group-level PET
  colocalization via a GUI).
- Questions: [NeuroStars, tag "nispace"](https://neurostars.org/tag/nispace); bugs and
  requests: [GitHub issues](https://github.com/LeonDLotter/NiSpace/issues).

## 12. Licenses and logo

Three separate licenses apply; tell the user which ones matter for their use.

- **Toolbox code:** [CC BY-NC-SA 4.0](LICENSE) (attribution, non-commercial, share-alike).
  This is not a permissive open-source license such as MIT: commercial use is not covered, and
  derived software must be shared under the same terms. For anything else, contact the author.
- **Reference data:** licensed separately. The processed maps and tables in the data
  repository are under CC BY-NC-SA 4.0, but individual datasets and even individual maps can
  have their own licenses. These are listed in the metadata and printed by `fetch_reference()`
  (per-map table via `nispace.datasets.fetch_map_info()`). The original sources' licenses
  apply in addition, and citing them is mandatory (section 4, rule 10).
- **Logo:** the files in [`logo/final/`](logo/final/) (e.g. `logo/final/nispace_logo.svg`,
  `logo/final/nispace_logo_fullname.svg`, `logo/final/nispace_icon.svg`) are under
  CC BY 4.0, unlike the rest of the repository. AI assistants may use them, e.g. in overview
  figures, slides or reports of analyses made with NiSpace, on two conditions:
  1. Give attribution: "NiSpace logo by Leon D. Lotter (https://github.com/LeonDLotter/NiSpace),
     CC BY 4.0", and indicate if the logo was changed.
  2. Cite NiSpace itself (see the
     [citation page](https://nispace.readthedocs.io/en/latest/citation.html)).

  The logo license grants no trademark rights: do not use the logo in a way that implies that
  NiSpace or its authors endorse a result, product or person. Details:
  [`logo/README.md`](logo/README.md).

## 13. If you change NiSpace

- Tests: `pytest` (unit tests); `pytest -m integration` needs the data repository and network.
- Every behaviour change needs a test; numerical methods are tested against established
  packages (scipy, statsmodels, etc.) where possible.
- Docstrings follow numpydoc and are the user documentation; update them with the code.
- `nispace/datalib/*.json` is generated from the data repository; do not edit by hand.
- This file is checked by `tests/test_agents_md.py`: every function name and path it mentions
  must exist.
