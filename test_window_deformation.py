"""Unit and edge-case tests for the von Karman window deformation solver."""

from __future__ import annotations

import numpy as np
import pytest

import window_deformation as wd


def _default_config(**overrides) -> wd.PlateConfig:
    params = {
        "diameter_mm": wd.DEFAULT_DIAMETER_MM,
        "thickness_mm": wd.DEFAULT_THICKNESS_MM,
        "pressure_pa": wd.DEFAULT_PRESSURE_PA,
    }
    params.update(overrides)
    return wd.PlateConfig.from_engineering_units(**params)


# --- Validation / edge cases ------------------------------------------------


@pytest.mark.parametrize("bad_value", [0.0, -1.0, -1e9])
def test_nonpositive_diameter_rejected(bad_value):
    with pytest.raises(ValueError, match="diameter_m"):
        wd.PlateConfig(diameter_m=bad_value, thickness_m=0.025, pressure_pa=1e5)


@pytest.mark.parametrize("bad_value", [0.0, -0.01])
def test_nonpositive_thickness_rejected(bad_value):
    with pytest.raises(ValueError, match="thickness_m"):
        wd.PlateConfig(diameter_m=0.45, thickness_m=bad_value, pressure_pa=1e5)


@pytest.mark.parametrize("bad_value", [0.0, -101000.0])
def test_nonpositive_pressure_rejected(bad_value):
    with pytest.raises(ValueError, match="pressure_pa"):
        wd.PlateConfig(diameter_m=0.45, thickness_m=0.025, pressure_pa=bad_value)


def test_invalid_boundary_condition_rejected():
    with pytest.raises(ValueError, match="boundary_condition"):
        wd.PlateConfig(
            diameter_m=0.45,
            thickness_m=0.025,
            pressure_pa=1e5,
            boundary_condition="floating",
        )


@pytest.mark.parametrize("nu", [-1.0, 0.5, 0.7, -1.2])
def test_poisson_ratio_out_of_range_rejected(nu):
    with pytest.raises(ValueError, match="poisson_ratio"):
        wd.Material(poisson_ratio=nu)


@pytest.mark.parametrize("modulus", [0.0, -1.0])
def test_nonpositive_modulus_rejected(modulus):
    with pytest.raises(ValueError, match="youngs_modulus_pa"):
        wd.Material(youngs_modulus_pa=modulus)


def test_nonpositive_allowable_stress_rejected():
    with pytest.raises(ValueError, match="allowable_stress_pa"):
        wd.Material(allowable_stress_pa=0.0)


def test_solve_plate_rejects_tiny_mesh():
    with pytest.raises(ValueError, match="n_mesh"):
        wd.solve_plate(_default_config(), n_mesh=3)


def test_solve_plate_rejects_tiny_output():
    with pytest.raises(ValueError, match="n_points"):
        wd.solve_plate(_default_config(), n_points=1)


# --- Physical correctness ---------------------------------------------------


def test_baseline_clamped_matches_known_result():
    solution = wd.solve_plate(_default_config(), n_points=500)
    assert solution.center_deflection_mm == pytest.approx(0.0262, abs=0.001)
    assert solution.max_tensile_mpa == pytest.approx(4.848, abs=0.1)
    assert solution.safety_factor == pytest.approx(0.967, abs=0.02)


def test_baseline_simply_supported_matches_known_result():
    config = _default_config(boundary_condition=wd.SIMPLY_SUPPORTED)
    solution = wd.solve_plate(config, n_points=500)
    assert solution.center_deflection_mm == pytest.approx(0.1156, abs=0.002)
    assert solution.max_tensile_mpa == pytest.approx(7.723, abs=0.1)


def test_nonlinear_matches_linear_for_thin_deflection():
    # w/t is tiny here, so the nonlinear solution must coincide with linear
    # plate theory; a mismatch indicates an ill-conditioned solver.
    solution = wd.solve_plate(_default_config(), n_points=500)
    assert solution.w0_nonlinear_m == pytest.approx(solution.w0_linear_m, rel=1e-3)


def test_edge_deflection_is_zero():
    solution = wd.solve_plate(_default_config(), n_points=300)
    assert solution.r_m[0] == 0.0
    assert solution.r_m[-1] == pytest.approx(_default_config().radius_m)
    assert abs(solution.w_m[-1]) < 1e-9


def test_center_is_maximum_deflection():
    solution = wd.solve_plate(_default_config(), n_points=300)
    assert np.argmax(solution.w_m) == 0
    assert solution.w0_nonlinear_m == pytest.approx(solution.w_m[0])


def test_deflection_monotonically_decreases_from_center():
    solution = wd.solve_plate(_default_config(), n_points=200)
    diffs = np.diff(solution.w_m)
    assert np.all(diffs <= 1e-12)


def test_simply_supported_deflects_more_than_clamped():
    clamped = wd.solve_plate(_default_config(boundary_condition=wd.CLAMPED))
    simply = wd.solve_plate(_default_config(boundary_condition=wd.SIMPLY_SUPPORTED))
    assert simply.center_deflection_mm > clamped.center_deflection_mm


def test_thicker_plate_deflects_less():
    thin = wd.solve_plate(_default_config(thickness_mm=15.0))
    thick = wd.solve_plate(_default_config(thickness_mm=35.0))
    assert thick.center_deflection_mm < thin.center_deflection_mm


def test_higher_pressure_increases_deflection_and_stress():
    low = wd.solve_plate(_default_config(pressure_pa=50_000.0))
    high = wd.solve_plate(_default_config(pressure_pa=150_000.0))
    assert high.center_deflection_mm > low.center_deflection_mm
    assert high.max_tensile_mpa > low.max_tensile_mpa


def test_small_pressure_approaches_linear_theory():
    # At very low load the nonlinear membrane term is negligible, so the
    # nonlinear centre deflection should approach the linear prediction.
    config = _default_config(pressure_pa=10.0)
    solution = wd.solve_plate(config)
    assert solution.w0_nonlinear_m == pytest.approx(solution.w0_linear_m, rel=1e-3)


def test_safety_factor_scales_with_allowable_stress():
    base_material = wd.Material(allowable_stress_pa=100.0 * wd.PA_PER_MPA)
    doubled_material = wd.Material(allowable_stress_pa=200.0 * wd.PA_PER_MPA)
    base_config = wd.PlateConfig.from_engineering_units(
        diameter_mm=wd.DEFAULT_DIAMETER_MM,
        thickness_mm=wd.DEFAULT_THICKNESS_MM,
        pressure_pa=wd.DEFAULT_PRESSURE_PA,
        material=base_material,
    )
    doubled_config = wd.PlateConfig.from_engineering_units(
        diameter_mm=wd.DEFAULT_DIAMETER_MM,
        thickness_mm=wd.DEFAULT_THICKNESS_MM,
        pressure_pa=wd.DEFAULT_PRESSURE_PA,
        material=doubled_material,
    )
    doubled = wd.solve_plate(doubled_config)
    base = wd.solve_plate(base_config)
    assert doubled.safety_factor == pytest.approx(2.0 * base.safety_factor, rel=0.02)


def test_linear_center_deflection_clamped_closed_form():
    config = _default_config()
    radius = config.radius_m
    expected = config.pressure_pa * radius**4 / (64.0 * config.flexural_rigidity)
    assert config.linear_center_deflection() == pytest.approx(expected)


# --- Design sweep -----------------------------------------------------------


def test_design_sweep_thickness_returns_sorted_points():
    points = wd.run_design_sweep(_default_config(), wd.THICKNESS_SWEEP, [15.0, 25.0, 35.0])
    assert [p.value for p in points] == [15.0, 25.0, 35.0]
    deflections = [p.center_deflection_mm for p in points]
    assert deflections[0] > deflections[1] > deflections[2]


def test_design_sweep_pressure_increases_stress():
    points = wd.run_design_sweep(
        _default_config(), wd.PRESSURE_SWEEP, [50_000.0, 150_000.0]
    )
    assert points[1].peak_stress_mpa > points[0].peak_stress_mpa


def test_design_sweep_diameter_increases_deflection():
    points = wd.run_design_sweep(_default_config(), wd.DIAMETER_SWEEP, [250.0, 350.0, 450.0])
    assert [p.value for p in points] == [250.0, 350.0, 450.0]
    deflections = [p.center_deflection_mm for p in points]
    assert deflections[0] < deflections[1] < deflections[2]


def test_design_sweep_rejects_bad_variable():
    with pytest.raises(ValueError, match="sweep_variable"):
        wd.run_design_sweep(_default_config(), "radius_mm", [1.0])


def test_design_sweep_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one"):
        wd.run_design_sweep(_default_config(), wd.THICKNESS_SWEEP, [])


# --- CLI plumbing -----------------------------------------------------------


def test_cli_sweep_requires_bounds():
    with pytest.raises(ValueError, match="sweep-start"):
        wd.main(["--sweep-variable", "thickness_mm", "--no-save"])


def test_config_from_args_uses_overrides():
    parser = wd.build_parser()
    args = parser.parse_args(
        [
            "--diameter-mm",
            "300",
            "--thickness-mm",
            "20",
            "--poisson-ratio",
            "0.22",
            "--boundary-condition",
            "simply_supported",
        ]
    )
    config = wd.config_from_args(args)
    assert config.diameter_m == pytest.approx(0.3)
    assert config.thickness_m == pytest.approx(0.02)
    assert config.material.poisson_ratio == pytest.approx(0.22)
    assert config.boundary_condition == wd.SIMPLY_SUPPORTED


def test_material_preset_selected_by_name():
    args = wd.build_parser().parse_args(["--material", "n_bk7"])
    config = wd.config_from_args(args)
    preset = wd.MATERIAL_PRESETS["n_bk7"]
    assert config.material.name == preset.name
    assert config.material.youngs_modulus_pa == pytest.approx(preset.youngs_modulus_pa)
    assert config.material.poisson_ratio == pytest.approx(preset.poisson_ratio)
    assert config.material.allowable_stress_pa == pytest.approx(preset.allowable_stress_pa)


def test_default_material_preset_is_fused_silica():
    config = wd.config_from_args(wd.build_parser().parse_args([]))
    fused = wd.MATERIAL_PRESETS["fused_silica"]
    assert config.material.name == fused.name
    assert config.material.youngs_modulus_pa == pytest.approx(fused.youngs_modulus_pa)
    assert config.material.poisson_ratio == pytest.approx(fused.poisson_ratio)


def test_property_flags_override_material_preset():
    args = wd.build_parser().parse_args(
        [
            "--material",
            "sapphire",
            "--poisson-ratio",
            "0.31",
            "--allowable-stress-mpa",
            "200",
        ]
    )
    config = wd.config_from_args(args)
    sapphire = wd.MATERIAL_PRESETS["sapphire"]
    assert config.material.poisson_ratio == pytest.approx(0.31)
    assert config.material.allowable_stress_pa == pytest.approx(200.0 * wd.PA_PER_MPA)
    assert config.material.youngs_modulus_pa == pytest.approx(sapphire.youngs_modulus_pa)
    assert config.material.name == sapphire.name


def test_invalid_material_preset_rejected():
    with pytest.raises(SystemExit):
        wd.build_parser().parse_args(["--material", "unobtainium"])


def test_table_i_glass_presets_available():
    # Table I glasses added from the reference glass-property tables, with the
    # allowable stress expressed as design stress (psi) = modulus of rupture / 10.
    expected_allowable_psi = {
        "aluminosilicate": 660.0,
        "vycor": 615.0,
        "pyrex": 610.0,
        "plate_glass": 650.0,
        "lead_glass": 500.0,
    }
    for key, allow_psi in expected_allowable_psi.items():
        assert key in wd.MATERIAL_PRESETS
        preset = wd.MATERIAL_PRESETS[key]
        assert preset.allowable_stress_pa == pytest.approx(allow_psi * wd.PA_PER_PSI)


def test_updated_allowable_design_stress_values():
    # Existing Table I glasses now use the table's allowable design stress (psi).
    assert wd.MATERIAL_PRESETS["fused_silica"].allowable_stress_pa == pytest.approx(
        680.0 * wd.PA_PER_PSI
    )
    assert wd.MATERIAL_PRESETS["n_bk7"].allowable_stress_pa == pytest.approx(
        500.0 * wd.PA_PER_PSI
    )


def test_acrylic_preset_available():
    # Methyl methacrylate (Acrylic) from Table I: E=360,000 psi, nu=0.39,
    # allowable design stress 920 psi.
    preset = wd.MATERIAL_PRESETS["acrylic"]
    assert preset.name == "Methyl Methacrylate"
    assert preset.youngs_modulus_pa == pytest.approx(360.0e3 * wd.PA_PER_PSI)
    assert preset.poisson_ratio == pytest.approx(0.39)
    assert preset.allowable_stress_pa == pytest.approx(920.0 * wd.PA_PER_PSI)


def test_acrylic_note8_helper_picks_worst_case():
    ref = wd.ACRYLIC_REFERENCE_MODULUS_PA
    # At the reference modulus both Note #8 options equal 920 psi.
    assert wd.acrylic_note8_allowable_stress_pa(ref) == pytest.approx(
        920.0 * wd.PA_PER_PSI
    )
    # Below the reference, option (b) scales down and becomes the worst case.
    assert wd.acrylic_note8_allowable_stress_pa(ref / 2.0) == pytest.approx(
        460.0 * wd.PA_PER_PSI
    )
    # Above the reference, option (a) = MoR/10 caps the allowable at 920 psi.
    assert wd.acrylic_note8_allowable_stress_pa(ref * 2.0) == pytest.approx(
        920.0 * wd.PA_PER_PSI
    )


def test_acrylic_default_allowable_is_nominal():
    config = wd.config_from_args(
        wd.build_parser().parse_args(["--material", "acrylic"])
    )
    assert config.material.allowable_stress_pa == pytest.approx(920.0 * wd.PA_PER_PSI)


def test_acrylic_low_modulus_override_lowers_allowable():
    # A modulus below the 360,000 psi reference must lower the allowable (and
    # thus the safety factor) instead of keeping the nominal 920 psi.
    ref = wd.ACRYLIC_REFERENCE_MODULUS_PA
    config = wd.config_from_args(
        wd.build_parser().parse_args(
            ["--material", "acrylic", "--youngs-modulus-pa", str(ref / 2.0)]
        )
    )
    assert config.material.allowable_stress_pa == pytest.approx(460.0 * wd.PA_PER_PSI)


def test_acrylic_high_modulus_override_capped_at_nominal():
    ref = wd.ACRYLIC_REFERENCE_MODULUS_PA
    config = wd.config_from_args(
        wd.build_parser().parse_args(
            ["--material", "acrylic", "--youngs-modulus-pa", str(ref * 2.0)]
        )
    )
    assert config.material.allowable_stress_pa == pytest.approx(920.0 * wd.PA_PER_PSI)


def test_acrylic_explicit_allowable_override_wins():
    ref = wd.ACRYLIC_REFERENCE_MODULUS_PA
    config = wd.config_from_args(
        wd.build_parser().parse_args(
            [
                "--material",
                "acrylic",
                "--youngs-modulus-pa",
                str(ref / 2.0),
                "--allowable-stress-mpa",
                "5",
            ]
        )
    )
    assert config.material.allowable_stress_pa == pytest.approx(5.0 * wd.PA_PER_MPA)


def test_list_materials_prints_table_and_skips_solve(capsys):
    wd.main(["--list-materials"])
    captured = capsys.readouterr()
    assert "Available material presets" in captured.out
    assert "sapphire" in captured.out
    assert "Nonlinear center deflection" not in captured.out


def test_material_preset_name_appears_in_console(capsys):
    wd.main(["--material", "sapphire", "--thickness-mm", "40", "--no-save"])
    captured = capsys.readouterr()
    assert "Sapphire Window Deformation" in captured.out


def test_main_single_case_runs_without_saving(capsys):
    wd.main(["--no-save"])
    captured = capsys.readouterr()
    assert "Nonlinear center deflection" in captured.out
    assert "Safety factor" in captured.out


def test_main_sweep_runs_without_saving(capsys):
    wd.main(
        [
            "--sweep-variable",
            "thickness_mm",
            "--sweep-start",
            "20",
            "--sweep-stop",
            "30",
            "--sweep-count",
            "3",
            "--no-save",
        ]
    )
    captured = capsys.readouterr()
    assert "Design Sweep Summary" in captured.out


def test_main_diameter_sweep_defaults_bounds(capsys):
    wd.main(["--sweep-variable", "diameter_mm", "--no-save"])
    captured = capsys.readouterr()
    assert "Design Sweep Summary" in captured.out
    assert "Diameter (mm)" in captured.out
    # Default diameter sweep is 250-450 mm in 5 steps.
    assert "250.000" in captured.out
    assert "450.000" in captured.out


def test_main_saves_jpeg_files(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--output-dir", str(out_dir)])
    files = sorted(p.name for p in out_dir.glob("*.jpg"))
    assert files == [
        "single_case_deflection_fused_silica_clamped.jpg",
        "single_case_stress_fused_silica_clamped.jpg",
    ]


def test_figure_format_exports_svg_alongside_jpg(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--figure-format", "jpg", "svg", "--output-dir", str(out_dir)])
    assert sorted(p.name for p in out_dir.glob("*.jpg")) == [
        "single_case_deflection_fused_silica_clamped.jpg",
        "single_case_stress_fused_silica_clamped.jpg",
    ]
    assert sorted(p.name for p in out_dir.glob("*.svg")) == [
        "single_case_deflection_fused_silica_clamped.svg",
        "single_case_stress_fused_silica_clamped.svg",
    ]


def test_figure_format_svg_only_writes_no_jpg(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--figure-format", "svg", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*.jpg"))
    assert sorted(p.name for p in out_dir.glob("*.svg")) == [
        "single_case_deflection_fused_silica_clamped.svg",
        "single_case_stress_fused_silica_clamped.svg",
    ]


def test_figure_format_multi_shares_unique_stem(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--figure-format", "jpg", "svg", "--output-dir", str(out_dir)])
    wd.main(["--figure-format", "jpg", "svg", "--output-dir", str(out_dir)])
    names = {p.name for p in out_dir.iterdir()}
    # The second run must not overwrite the first; both formats bump together.
    assert "single_case_deflection_fused_silica_clamped_1.jpg" in names
    assert "single_case_deflection_fused_silica_clamped_1.svg" in names


def test_invalid_figure_format_rejected():
    with pytest.raises(SystemExit):
        wd.build_parser().parse_args(["--figure-format", "gif"])


def test_existing_files_are_not_overwritten(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--no-excel", "--output-dir", str(out_dir)])
    first = {p.name: p.stat().st_mtime_ns for p in out_dir.glob("*.jpg")}

    wd.main(["--no-excel", "--output-dir", str(out_dir)])
    after = {p.name for p in out_dir.glob("*.jpg")}

    # The boundary condition is always appended (default is clamped), and the
    # original files remain untouched while a numeric-suffixed copy is added.
    for name, mtime in first.items():
        assert (out_dir / name).stat().st_mtime_ns == mtime
    assert "single_case_deflection_fused_silica_clamped.jpg" in first
    assert "single_case_stress_fused_silica_clamped.jpg" in first
    assert "single_case_deflection_fused_silica_clamped_1.jpg" in after
    assert "single_case_stress_fused_silica_clamped_1.jpg" in after


def test_unique_path_always_appends_boundary_condition(tmp_path):
    path = wd._unique_path(str(tmp_path), "sweep_combined", "xlsx", "clamped")
    assert path.endswith("sweep_combined_clamped.xlsx")


def test_unique_path_falls_back_to_numeric_after_boundary_condition(tmp_path):
    (tmp_path / "sweep_combined_clamped.xlsx").write_text("existing")
    path = wd._unique_path(str(tmp_path), "sweep_combined", "xlsx", "clamped")
    assert path.endswith("sweep_combined_clamped_1.xlsx")


def test_unique_path_numeric_without_boundary_condition(tmp_path):
    (tmp_path / "sweep_combined.xlsx").write_text("existing")
    path = wd._unique_path(str(tmp_path), "sweep_combined", "xlsx")
    assert path.endswith("sweep_combined_1.xlsx")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Fused Silica", "fused_silica"),
        ("N-BK7", "n_bk7"),
        ("Methyl Methacrylate", "methyl_methacrylate"),
        ("96% Silica (Vycor)", "96_silica_vycor"),
        ("Borosilicate (Pyrex)", "borosilicate_pyrex"),
        ("", None),
        (None, None),
    ],
)
def test_material_slug(name, expected):
    assert wd._material_slug(name) == expected


def test_unique_path_inserts_material_before_boundary(tmp_path):
    path = wd._unique_path(
        str(tmp_path), "single_case", "xlsx", "clamped", "methyl_methacrylate"
    )
    assert path.endswith("single_case_methyl_methacrylate_clamped.xlsx")


def test_main_uses_selected_material_in_filenames(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--material", "acrylic", "--output-dir", str(out_dir)])
    assert (out_dir / "single_case_methyl_methacrylate_clamped.xlsx").is_file()
    assert sorted(p.name for p in out_dir.glob("*.jpg")) == [
        "single_case_deflection_methyl_methacrylate_clamped.jpg",
        "single_case_stress_methyl_methacrylate_clamped.jpg",
    ]


def test_main_sweep_saves_jpeg_file(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(
        [
            "--sweep-variable",
            "diameter_mm",
            "--output-dir",
            str(out_dir),
        ]
    )
    files = [p.name for p in out_dir.glob("*.jpg")]
    assert "sweep_diameter_mm_fused_silica_clamped.jpg" in files


def test_main_combined_sweep_prints_grid_matrices(capsys):
    wd.main(["--sweep-variable", "all", "--no-save"])
    captured = capsys.readouterr()
    assert "Center Deflection (mm)" in captured.out
    assert "Peak Tensile Stress (MPa)" in captured.out
    assert "Safety Factor" in captured.out
    assert "Rows: Thickness (mm)  |  Columns: Diameter (mm)" in captured.out
    # Default grid axes: thickness 15-35 mm, diameter 250-450 mm.
    assert "15.0" in captured.out
    assert "35.0" in captured.out
    assert "250.0" in captured.out
    assert "450.0" in captured.out


def test_main_combined_sweep_warns_on_ignored_single_sweep_flags(capsys):
    wd.main(["--sweep-variable", "all", "--sweep-start", "10", "--no-save"])
    captured = capsys.readouterr()
    assert "IGNORED ARGUMENTS WARNING" in captured.err
    assert "--sweep-start" in captured.err
    # Points the user at the correct grid flags.
    assert "--thickness-sweep-start" in captured.err
    # The grid still runs after the warning.
    assert "Center Deflection (mm)" in captured.out


def test_main_combined_sweep_no_warning_without_single_sweep_flags(capsys):
    wd.main(
        [
            "--sweep-variable",
            "all",
            "--thickness-sweep-stop",
            "60",
            "--no-save",
        ]
    )
    captured = capsys.readouterr()
    assert "IGNORED ARGUMENTS WARNING" not in captured.err


def test_main_combined_sweep_saves_single_figure(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--output-dir", str(out_dir)])
    files = [p.name for p in out_dir.glob("*.jpg")]
    assert files == ["sweep_combined_fused_silica_clamped.jpg"]


def test_main_single_case_saves_excel(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--output-dir", str(out_dir)])
    assert (out_dir / "single_case_fused_silica_clamped.xlsx").is_file()


def test_single_case_excel_has_profile_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    config = _default_config()
    solution = wd.solve_plate(config, n_points=60)
    paths = wd._export_single_case_excel(config, solution, str(tmp_path))
    workbook = openpyxl.load_workbook(paths[0])
    assert "Profile Data" in workbook.sheetnames
    sheet = workbook["Profile Data"]
    header = [cell.value for cell in sheet[1]]
    assert header == [
        "Radius (mm)",
        "Deflection (mm)",
        "Sigma_r top (MPa)",
        "Sigma_theta top (MPa)",
        "Sigma_r bottom (MPa)",
        "Sigma_theta bottom (MPa)",
    ]
    # One data row per solver sample point plus the header row.
    assert sheet.max_row == solution.r_m.size + 1
    first_data = [cell.value for cell in sheet[2]]
    assert first_data[0] == pytest.approx(solution.r_m[0] * wd.MM_PER_M)
    assert first_data[1] == pytest.approx(solution.w_m[0] * wd.MM_PER_M)


def test_single_case_excel_includes_material_row(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    config = _default_config(material=wd.MATERIAL_PRESETS["acrylic"])
    solution = wd.solve_plate(config, n_points=60)
    paths = wd._export_single_case_excel(config, solution, str(tmp_path))
    sheet = openpyxl.load_workbook(paths[0])["Single Case"]
    material_value = None
    for row in sheet.iter_rows(values_only=True):
        if row and row[0] == "Material":
            material_value = row[1]
            break
    assert material_value == "Methyl Methacrylate"


def test_held_constant_label_includes_material():
    config = _default_config()
    label = wd._held_constant_label(config, wd.THICKNESS_SWEEP)
    assert "Material = Fused Silica" in label


def test_combined_sweep_case_details_includes_material(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--output-dir", str(out_dir)])
    workbook = openpyxl.load_workbook(
        out_dir / "sweep_combined_fused_silica_clamped.xlsx"
    )
    sheet = workbook["Case Details"]
    rows = list(sheet.iter_rows(values_only=True))
    material_row = next((r for r in rows if r and r[0] == "Material"), None)
    assert material_row is not None
    assert material_row[1] == "Fused Silica"



    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "diameter_mm", "--output-dir", str(out_dir)])
    assert (out_dir / "sweep_diameter_mm_fused_silica_clamped.xlsx").is_file()


def test_custom_names_single_case(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(
        [
            "--figure-name",
            "mywindow",
            "--excel-name",
            "mydata",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert (out_dir / "mywindow_deflection_fused_silica_clamped.jpg").is_file()
    assert (out_dir / "mywindow_stress_fused_silica_clamped.jpg").is_file()
    assert (out_dir / "mydata_fused_silica_clamped.xlsx").is_file()
    # Default-named files must not be produced when custom names are given.
    assert not (out_dir / "single_case_deflection_fused_silica_clamped.jpg").is_file()
    assert not (out_dir / "single_case_fused_silica_clamped.xlsx").is_file()


def test_custom_names_single_sweep(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(
        [
            "--sweep-variable",
            "thickness_mm",
            "--sweep-start",
            "15",
            "--sweep-stop",
            "35",
            "--sweep-count",
            "3",
            "--figure-name",
            "mysweep",
            "--excel-name",
            "mysweepdata",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert (out_dir / "mysweep_fused_silica_clamped.jpg").is_file()
    assert (out_dir / "mysweepdata_fused_silica_clamped.xlsx").is_file()


def test_custom_names_combined_sweep_keep_boundary_suffix(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(
        [
            "--sweep-variable",
            "all",
            "--figure-name",
            "mygrid",
            "--excel-name",
            "mygriddata",
            "--boundary-condition",
            "simply_supported",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert (out_dir / "mygrid_fused_silica_simply_supported.jpg").is_file()
    assert (out_dir / "mygriddata_fused_silica_simply_supported.xlsx").is_file()


def test_main_combined_sweep_saves_excel_with_three_sheets(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--output-dir", str(out_dir)])
    workbook_path = out_dir / "sweep_combined_fused_silica_clamped.xlsx"
    assert workbook_path.is_file()

    workbook = openpyxl.load_workbook(workbook_path)
    assert workbook.sheetnames == [
        "Case Details",
        "Center Deflection",
        "Peak Tensile Stress",
        "Safety Factor",
    ]
    sheet = workbook["Center Deflection"]
    # The held-constant parameter block appears above the matrix.
    column_a = [sheet.cell(row=r, column=1).value for r in range(1, sheet.max_row + 1)]
    assert "Parameters held constant" in column_a
    assert "Pressure" in column_a
    assert "Boundary condition" in column_a
    assert "Young's modulus" in column_a
    assert "Poisson's ratio" in column_a
    assert "Allowable stress" in column_a

    header_row = column_a.index("Thickness (mm) \\ Diameter (mm)") + 1
    header = [cell.value for cell in sheet[header_row]]
    assert header == ["Thickness (mm) \\ Diameter (mm)", 250, 300, 350, 400, 450]
    # First data row is the 15 mm thickness row; column B is the 250 mm case.
    # 15/250 -> t/D = 0.06 (5 < R/h <= 10) -> auto resolves to Mindlin.
    data_row = header_row + 1
    assert sheet.cell(row=data_row, column=1).value == 15
    assert sheet.cell(row=data_row, column=2).value == pytest.approx(0.01950, abs=1e-4)


def test_combined_sweep_case_details_sheet_has_all_single_case_data(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--output-dir", str(out_dir)])
    workbook = openpyxl.load_workbook(out_dir / "sweep_combined_fused_silica_clamped.xlsx")

    sheet = workbook["Case Details"]
    column_a = [sheet.cell(row=r, column=1).value for r in range(1, sheet.max_row + 1)]
    # Held-constant material/load block carries over from the single-case sheet.
    for label in (
        "Parameters held constant",
        "Pressure",
        "Boundary condition",
        "Young's modulus",
        "Poisson's ratio",
        "Allowable stress",
        "Per-case results",
    ):
        assert label in column_a

    header_row = column_a.index("Thickness (mm)") + 1
    header = [cell.value for cell in sheet[header_row]]
    assert header == [
        "Thickness (mm)",
        "Diameter (mm)",
        "Thickness/diameter ratio",
        "Plate theory",
        "Linear center deflection (mm)",
        "Nonlinear center deflection (mm)",
        "Peak tensile stress (MPa)",
        "Safety factor",
    ]

    # 5 thickness x 5 diameter = 25 per-case rows follow the header.
    data_rows = [
        [sheet.cell(row=r, column=c).value for c in range(1, 9)]
        for r in range(header_row + 1, header_row + 1 + 25)
    ]
    assert len(data_rows) == 25
    first = data_rows[0]
    assert first[0] == pytest.approx(15.0)  # thickness
    assert first[1] == pytest.approx(250.0)  # diameter
    assert first[2] == pytest.approx(15.0 / 250.0)  # t/D ratio
    assert first[3] == "Mindlin"  # resolved theory (t/D = 0.06 -> R/h < 10)
    assert first[4] is not None and first[4] > 0  # linear deflection present
    assert first[5] is not None and first[5] > 0  # nonlinear deflection present


@pytest.mark.parametrize(
    ("args", "filename", "sheet_name"),
    [
        ([], "single_case_fused_silica_clamped.xlsx", "Single Case"),
        (
            ["--sweep-variable", "thickness_mm", "--sweep-start", "15",
             "--sweep-stop", "35", "--sweep-count", "5"],
            "sweep_thickness_mm_fused_silica_clamped.xlsx",
            "Design Sweep",
        ),
        (["--sweep-variable", "all"], "sweep_combined_fused_silica_clamped.xlsx", "Case Details"),
    ],
)
def test_excel_includes_plate_theory_threshold_note(tmp_path, args, filename, sheet_name):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "figs"
    wd.main([*args, "--output-dir", str(out_dir)])
    sheet = openpyxl.load_workbook(out_dir / filename)[sheet_name]

    note = None
    for row in sheet.iter_rows(values_only=True):
        if row and row[0] == "Note" and row[1] and "Plate theory selection" in row[1]:
            note = row[1]
            break

    assert note is not None
    thin_limit = f"{wd.MINDLIN_AUTO_SWITCH_RATIO:g}"
    assert f"t/D < {thin_limit}" in note
    assert f"t/D >= {thin_limit}" in note
    assert "R/h" in note
    assert "Kirchhoff" in note
    assert "Mindlin" in note


@pytest.mark.parametrize(
    ("args", "filename", "sheet_name", "both_columns"),
    [
        ([], "single_case_fused_silica_clamped.xlsx", "Single Case", True),
        (
            ["--sweep-variable", "thickness_mm", "--sweep-start", "15",
             "--sweep-stop", "35", "--sweep-count", "5"],
            "sweep_thickness_mm_fused_silica_clamped.xlsx",
            "Design Sweep",
            False,
        ),
        (
            ["--sweep-variable", "all"],
            "sweep_combined_fused_silica_clamped.xlsx",
            "Case Details",
            True,
        ),
    ],
)
def test_excel_includes_deflection_guidance_note(
    tmp_path, args, filename, sheet_name, both_columns
):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "figs"
    wd.main([*args, "--output-dir", str(out_dir)])
    sheet = openpyxl.load_workbook(out_dir / filename)[sheet_name]

    note = None
    for row in sheet.iter_rows(values_only=True):
        if row and row[0] == "Note" and row[1] and "deflection" in row[1].lower():
            if "Plate theory selection" in row[1]:
                continue
            note = row[1]
            break

    assert note is not None
    assert "NONLINEAR" in note
    assert "w0/t" in note
    if both_columns:
        assert "Linear center deflection" in note


def test_no_save_skips_excel(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--no-save", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*.xlsx"))


def test_no_save_skips_all_files(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--no-save", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*"))


def test_no_excel_keeps_figures(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--no-excel", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*.xlsx"))
    assert sorted(p.name for p in out_dir.glob("*.jpg")) == [
        "single_case_deflection_fused_silica_clamped.jpg",
        "single_case_stress_fused_silica_clamped.jpg",
    ]


def test_no_figures_keeps_excel(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--no-figures", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*.jpg"))
    assert (out_dir / "single_case_fused_silica_clamped.xlsx").is_file()


def test_no_excel_combined_keeps_figure(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--no-excel", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*.xlsx"))
    assert (out_dir / "sweep_combined_fused_silica_clamped.jpg").is_file()


def test_no_figures_combined_keeps_excel(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(["--sweep-variable", "all", "--no-figures", "--output-dir", str(out_dir)])
    assert not list(out_dir.glob("*.jpg"))
    assert (out_dir / "sweep_combined_fused_silica_clamped.xlsx").is_file()


def test_held_constant_label_excludes_swept_variable():
    config = _default_config()
    label = wd._held_constant_label(config, wd.THICKNESS_SWEEP)
    assert "Thickness" not in label
    assert "Diameter = 400.0 mm" in label
    assert "Pressure = 101,000 Pa" in label
    assert "Edge = clamped" in label


def test_held_constant_label_diameter_sweep_excludes_diameter():
    config = _default_config()
    label = wd._held_constant_label(config, wd.DIAMETER_SWEEP)
    assert "Diameter" not in label
    assert "Thickness = 25.0 mm" in label


def test_run_sweep_grid_covers_all_combinations():
    config = _default_config()
    points = wd.run_sweep_grid(config, [15.0, 25.0], [250.0, 450.0])
    assert len(points) == 4
    combos = {(p.thickness_mm, p.diameter_mm) for p in points}
    assert combos == {(15.0, 250.0), (15.0, 450.0), (25.0, 250.0), (25.0, 450.0)}


def test_run_sweep_grid_thicker_plate_deflects_less():
    config = _default_config()
    points = wd.run_sweep_grid(config, [15.0, 35.0], [450.0])
    by_thickness = {p.thickness_mm: p.center_deflection_mm for p in points}
    assert by_thickness[35.0] < by_thickness[15.0]


def test_run_sweep_grid_rejects_empty_axis():
    config = _default_config()
    with pytest.raises(ValueError):
        wd.run_sweep_grid(config, [], [450.0])


# --- Mindlin-Reissner plate theory -----------------------------------------


def _thick_config(thickness_mm, diameter_mm=250.0, **overrides):
    params = {
        "diameter_mm": diameter_mm,
        "thickness_mm": thickness_mm,
        "pressure_pa": wd.DEFAULT_PRESSURE_PA,
    }
    params.update(overrides)
    return wd.PlateConfig.from_engineering_units(**params)


def test_display_plate_theory_capitalizes_names():
    assert wd._display_plate_theory(wd.KIRCHHOFF) == "Kirchhoff"
    assert wd._display_plate_theory(wd.MINDLIN) == "Mindlin"
    # Non-theory tokens (e.g. auto) are returned unchanged.
    assert wd._display_plate_theory(wd.AUTO) == wd.AUTO


def test_single_case_excel_plate_theory_is_capitalized(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "figs"
    wd.main(["--plate-theory", "mindlin", "--output-dir", str(out_dir)])
    sheet = openpyxl.load_workbook(
        out_dir / "single_case_fused_silica_clamped.xlsx"
    )["Single Case"]
    theory_value = None
    for row in sheet.iter_rows(values_only=True):
        if row and row[0] == "Plate theory":
            theory_value = row[1]
            break
    assert theory_value == "Mindlin"


def test_resolved_plate_theory_explicit_choices():
    thin = _default_config(plate_theory=wd.KIRCHHOFF)
    thick = _thick_config(35.0, plate_theory=wd.MINDLIN)
    assert thin.resolved_plate_theory() == wd.KIRCHHOFF
    assert thick.resolved_plate_theory() == wd.MINDLIN


def test_resolved_plate_theory_auto_switches_on_ratio():
    # Thin: 10 mm on a 450 mm window -> t/D = 0.022 < 0.05 (R/h > 10) -> Kirchhoff.
    thin = _thick_config(10.0, diameter_mm=450.0, plate_theory=wd.AUTO)
    assert thin.thickness_to_diameter_ratio < wd.MINDLIN_AUTO_SWITCH_RATIO
    assert thin.resolved_plate_theory() == wd.KIRCHHOFF
    # Transition: default 25/450 -> t/D = 0.056 (5 < R/h <= 10) -> Mindlin.
    transition = _default_config(plate_theory=wd.AUTO)
    assert (
        wd.MINDLIN_AUTO_SWITCH_RATIO
        <= transition.thickness_to_diameter_ratio
        < wd.MINDLIN_THICKNESS_RATIO_THRESHOLD
    )
    assert transition.resolved_plate_theory() == wd.MINDLIN
    # Thick: 35 mm on a 250 mm window -> t/D = 0.14 > 0.1 (R/h < 5) -> Mindlin.
    thick = _thick_config(35.0, plate_theory=wd.AUTO)
    assert thick.thickness_to_diameter_ratio > wd.MINDLIN_THICKNESS_RATIO_THRESHOLD
    assert thick.resolved_plate_theory() == wd.MINDLIN


@pytest.mark.parametrize(
    ("thickness_mm", "diameter_mm"),
    [(15.0, 300.0), (20.0, 400.0), (25.0, 500.0)],
)
def test_resolved_plate_theory_at_auto_switch_boundary_is_mindlin(
    thickness_mm, diameter_mm
):
    # t/D exactly at the 0.05 auto-switch limit (R/h = 10) is inclusive -> Mindlin,
    # regardless of the floating-point noise in the metre-based ratio.
    config = wd.PlateConfig(
        diameter_m=diameter_mm / wd.MM_PER_M,
        thickness_m=thickness_mm / wd.MM_PER_M,
        pressure_pa=101_000.0,
        plate_theory=wd.AUTO,
    )
    assert config.thickness_to_diameter_ratio == pytest.approx(
        wd.MINDLIN_AUTO_SWITCH_RATIO
    )
    assert config.resolved_plate_theory() == wd.MINDLIN


def test_resolved_plate_theory_just_below_auto_switch_is_kirchhoff():
    # t/D = 0.049 (R/h ~ 10.2) sits just below the switch -> Kirchhoff.
    config = wd.PlateConfig(
        diameter_m=0.450,
        thickness_m=0.049 * 0.450,
        pressure_pa=101_000.0,
        plate_theory=wd.AUTO,
    )
    assert config.thickness_to_diameter_ratio < wd.MINDLIN_AUTO_SWITCH_RATIO
    assert config.resolved_plate_theory() == wd.KIRCHHOFF



    with pytest.raises(ValueError, match="plate_theory"):
        wd.PlateConfig(
            diameter_m=0.45,
            thickness_m=0.025,
            pressure_pa=1e5,
            plate_theory="bogus",
        )


def test_solution_records_plate_theory():
    config = _thick_config(35.0, plate_theory=wd.MINDLIN)
    solution = wd.solve_plate(config)
    assert solution.plate_theory == wd.MINDLIN


def test_auto_solution_records_resolved_theory():
    thin = wd.solve_plate(_thick_config(10.0, diameter_mm=450.0, plate_theory=wd.AUTO))
    thick = wd.solve_plate(_thick_config(35.0, plate_theory=wd.AUTO))
    assert thin.plate_theory == wd.KIRCHHOFF
    assert thick.plate_theory == wd.MINDLIN


def test_mindlin_thin_limit_matches_kirchhoff():
    # A thin plate has negligible shear, so Mindlin ~= the analytic Kirchhoff
    # (pure-bending) deflection. Compare against the closed form because the
    # nonlinear Kirchhoff BVP is not needed to validate the shear-free limit.
    config = _thick_config(10.0, diameter_mm=450.0, plate_theory=wd.MINDLIN)
    expected_mm = config.linear_center_deflection() * wd.MM_PER_M
    mindlin = wd.solve_plate(config)
    assert mindlin.center_deflection_mm == pytest.approx(expected_mm, rel=0.02)


def test_mindlin_clamped_matches_analytic_shear_deflection():
    # Clamped linear target: w = p a^4/(64 D) + p a^2/(4 kappa G t).
    config = _thick_config(35.0, diameter_mm=250.0, plate_theory=wd.MINDLIN)
    a = config.radius_m
    bending = config.pressure_pa * a**4 / (64.0 * config.flexural_rigidity)
    shear = config.pressure_pa * a**2 / (4.0 * config.shear_stiffness)
    expected_mm = (bending + shear) * wd.MM_PER_M
    solution = wd.solve_plate(config)
    assert solution.center_deflection_mm == pytest.approx(expected_mm, rel=0.03)


def test_mindlin_thick_deflects_more_than_kirchhoff():
    # Shear compliance makes a thick plate deflect more than the Kirchhoff model.
    kirchhoff = wd.solve_plate(_thick_config(35.0, diameter_mm=250.0))
    mindlin = wd.solve_plate(
        _thick_config(35.0, diameter_mm=250.0, plate_theory=wd.MINDLIN)
    )
    assert mindlin.center_deflection_mm > kirchhoff.center_deflection_mm


def test_mindlin_simply_supported_converges():
    config = _thick_config(
        35.0,
        diameter_mm=250.0,
        boundary_condition=wd.SIMPLY_SUPPORTED,
        plate_theory=wd.MINDLIN,
    )
    solution = wd.solve_plate(config)
    assert solution.plate_theory == wd.MINDLIN
    assert solution.center_deflection_mm > 0.0


def test_cli_accepts_plate_theory(tmp_path):
    out_dir = tmp_path / "figs"
    wd.main(
        [
            "--plate-theory",
            "mindlin",
            "--no-figures",
            "--no-excel",
            "--output-dir",
            str(out_dir),
        ]
    )


def test_cli_rejects_invalid_plate_theory():
    with pytest.raises(SystemExit):
        wd.build_parser().parse_args(["--plate-theory", "bogus"])


def test_config_from_args_default_plate_theory_is_auto():
    args = wd.build_parser().parse_args([])
    config = wd.config_from_args(args)
    assert config.plate_theory == wd.AUTO


# --- Thin-plate solver robustness (continuation + graceful fallback) --------


def test_linear_plate_fields_clamped_matches_closed_form():
    config = _default_config()
    r = np.linspace(0.0, config.radius_m, 200)
    _, w, m_r, m_t, n_r, n_t = wd._linear_plate_fields(config, r)
    # Centre deflection equals the closed-form linear value; edges vanish.
    assert w[0] == pytest.approx(config.linear_center_deflection(), rel=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    # Linear theory carries no membrane resultants.
    assert np.allclose(n_r, 0.0)
    assert np.allclose(n_t, 0.0)
    # At the centre M_r = M_t (axisymmetric bending).
    assert m_r[0] == pytest.approx(m_t[0], rel=1e-12)


def test_linear_plate_fields_simply_supported_matches_closed_form():
    config = _default_config(boundary_condition=wd.SIMPLY_SUPPORTED)
    r = np.linspace(0.0, config.radius_m, 200)
    _, w, m_r, _, _, _ = wd._linear_plate_fields(config, r)
    assert w[0] == pytest.approx(config.linear_center_deflection(), rel=1e-12)
    # Simply supported edge carries no radial moment.
    assert m_r[-1] == pytest.approx(0.0, abs=1e-9)


def test_linear_fallback_matches_bvp_in_linear_regime():
    # In the near-linear regime the closed-form fallback matches the BVP.
    config = _default_config()
    r = np.linspace(0.0, config.radius_m, wd.DEFAULT_OUTPUT_POINTS)
    _, w_lin, *_ = wd._linear_plate_fields(config, r)
    bvp = wd.solve_plate(config)
    assert w_lin[0] * wd.MM_PER_M == pytest.approx(bvp.center_deflection_mm, rel=1e-3)


def test_kirchhoff_fallback_uses_linear_solution_when_near_linear(monkeypatch):
    # Force the nonlinear solver to fail; a near-linear case must fall back to
    # the accurate closed-form solution (with a warning) rather than raise.
    class _Failed:
        success = False
        message = "forced failure"

    monkeypatch.setattr(wd, "solve_bvp", lambda *a, **k: _Failed())
    config = _default_config()  # w0/t ~ 0.0017, deep in the linear regime
    with pytest.warns(UserWarning, match="closed-form linear"):
        solution = wd.solve_plate(config)
    assert solution.center_deflection_mm == pytest.approx(
        config.linear_center_deflection() * wd.MM_PER_M, rel=1e-9
    )


def test_kirchhoff_raises_when_nonlinear_and_not_convergent(monkeypatch):
    # Force failure on a strongly nonlinear (large w0/t) case: no silent wrong
    # fallback is allowed, so an informative error must be raised.
    class _Failed:
        success = False
        message = "forced failure"

    monkeypatch.setattr(wd, "solve_bvp", lambda *a, **k: _Failed())
    config = _thick_config(4.0, diameter_mm=450.0, plate_theory=wd.KIRCHHOFF)
    with pytest.raises(RuntimeError, match="failed to converge"):
        wd.solve_plate(config)


def test_von_karman_validity_warning_thresholds():
    assert wd._von_karman_validity_warning(0.5) is None
    assert wd._von_karman_validity_warning(wd.VON_KARMAN_WT_WARN_RATIO) is None
    message = wd._von_karman_validity_warning(wd.VON_KARMAN_WT_WARN_RATIO + 1.0)
    assert message is not None
    assert "von Karman" in message


def test_solve_plate_warns_beyond_von_karman(monkeypatch):
    # A solution with large w0/t must trigger the validity warning.
    def _fake_fields(config, n_mesh, n_points):
        r = np.linspace(0.0, config.radius_m, n_points)
        w = np.full_like(r, 10.0 * config.thickness_m)  # w0/t = 10
        zeros = np.zeros_like(r)
        return r, w, zeros, zeros, zeros, zeros

    monkeypatch.setattr(wd, "_solve_kirchhoff_fields", _fake_fields)
    config = _default_config(plate_theory=wd.KIRCHHOFF)
    with pytest.warns(UserWarning, match="exceeds the"):
        wd.solve_plate(config)


def _invalid_solution(config):
    """Build a PlateSolution flagged as beyond von Karman validity (w0/t = 12)."""
    r = np.linspace(0.0, config.radius_m, 50)
    w = 12.0 * config.thickness_m * (1.0 - (r / config.radius_m) ** 2)
    zeros = np.zeros_like(r)
    ones = zeros + 1.0e5
    return wd._assemble_solution(
        config, wd.KIRCHHOFF, r, w, ones, ones, zeros, zeros
    )


def test_solution_records_validity_warning():
    config = _default_config(plate_theory=wd.KIRCHHOFF)
    solution = _invalid_solution(config)
    assert solution.exceeds_von_karman_validity
    assert solution.validity_warning is not None


def test_print_single_case_shows_validity_banner(capsys):
    config = _default_config(plate_theory=wd.KIRCHHOFF)
    solution = _invalid_solution(config)
    wd._print_single_case(config, solution)
    captured = capsys.readouterr()
    # The unmissable banner goes to stderr.
    assert "MODEL VALIDITY WARNING" in captured.err
    assert captured.err.count("MODEL VALIDITY WARNING") >= 2  # top and bottom


def test_print_sweep_flags_invalid_points(capsys):
    points = [
        wd.SweepPoint(10.0, 0.1, 1.0, 5.0, exceeds_von_karman=False),
        wd.SweepPoint(5.0, 50.0, 100.0, 0.1, exceeds_von_karman=True),
    ]
    wd._print_sweep(wd.THICKNESS_SWEEP, points)
    captured = capsys.readouterr()
    assert "EXCEEDS von Karman" in captured.out
    assert "MODEL VALIDITY WARNING" in captured.err


def test_single_case_excel_has_red_validity_row(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    config = _default_config(plate_theory=wd.KIRCHHOFF)
    solution = _invalid_solution(config)
    paths = wd._export_single_case_excel(config, solution, str(tmp_path))
    workbook = openpyxl.load_workbook(paths[0])
    sheet = workbook.active
    flagged = [
        cell
        for row in sheet.iter_rows()
        for cell in row
        if cell.value and "MODEL VALIDITY WARNING" in str(cell.value)
    ]
    assert flagged
    assert flagged[0].fill.fgColor.rgb == "FFC00000"


def test_valid_case_has_no_validity_surfacing(capsys, tmp_path):
    # A normal thin case must not emit any validity banner anywhere.
    config = _default_config()
    solution = wd.solve_plate(config)
    assert not solution.exceeds_von_karman_validity
    wd._print_single_case(config, solution)
    captured = capsys.readouterr()
    assert "MODEL VALIDITY WARNING" not in captured.err
    assert "MODEL VALIDITY WARNING" not in captured.out

