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
