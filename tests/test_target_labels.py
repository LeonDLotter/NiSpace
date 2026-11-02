"""plotting.nice_target_label(): display names for PET targets (exact-name lookup)."""

from nispace.plotting import nice_target_label


def test_known_targets_get_subscripts():
    assert nice_target_label("GABAa") == r"GABA$_\mathrm{A}$"
    assert nice_target_label("5HT1a") == r"5HT$_\mathrm{1A}$"
    assert nice_target_label("5HT2a") == r"5HT$_\mathrm{2A}$"
    assert nice_target_label("D23") == r"D$_\mathrm{2/3}$"


def test_other_names_unchanged():
    for name in ["mGluR5", "5HTT", "rCPS", "COX1", "CMRglu", "c0", "Dopamine", "D2/3x"]:
        assert nice_target_label(name) == name
    assert nice_target_label(3) == "3"


def test_greek_letters_are_text_and_bold_math():
    # Greek letters as text (bold with the label); math subscripts made bold explicitly
    from nispace.plotting import _bold_math
    assert "$" not in nice_target_label("GABAa5").split("$")[-1]   # "α5" outside math
    assert nice_target_label("A4B2") == r"α$_\mathrm{4}$β$_\mathrm{2}$"
    assert _bold_math(nice_target_label("5HT2a")) == r"5HT$_\mathbf{2A}$"
    assert _bold_math("Dopamine $(n=3)$") == "Dopamine $(n=3)$"


def test_pet_map_label():
    from nispace.plotting import _pet_map_label
    assert _pet_map_label("target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015") == \
        r"D$_\mathrm{2/3}$ $(\mathrm{Sandiego2015},\ n=55)$"
    assert _pet_map_label("Dopamine") is None
