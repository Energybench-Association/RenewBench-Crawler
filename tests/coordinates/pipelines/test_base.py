# tests/coordinates/pipelines/test_base.py
"""Structural tests for BasePipeline's shared scaffolding (not pipeline-specific steps)."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

from rbc.coordinates.load import Run
from rbc.coordinates.mappings import (
    OPERATOR_COLUMNS,
    OPERATOR_METADATA,
    SYSOP_CODE_COL,
    SYSOP_FUEL_COL,
    SYSOP_FUEL_SUB_COL,
    SYSOP_NAME_COL,
    SYSOP_REGION_COL,
    OperatorInfo,
)
from rbc.coordinates.match_schema import MatchCandidate
from rbc.coordinates.pipelines._base import BasePipeline
from rbc.coordinates.resources.gem import GEMLocator
from rbc.coordinates.resources.natural_earth import RegionRegistry
from rbc.coordinates.resources.osmpp import OSMPPLocator

BASE_MODULE = "rbc.coordinates.pipelines._base"


# ----------------------------------
# Fixtures / test doubles
# ----------------------------------
@pytest.fixture
def eia_run(make_run: Callable[..., Run]) -> Run:
    """An "eia" run (no rows needed for structural tests).

    Args:
        make_run (Callable[..., Run]): Factory for runs (s. conftest).

    Returns:
        Run: An (empty) EIA run.
    """
    return make_run("eia")


@pytest.fixture
def cen_rows() -> pd.DataFrame:
    """Raw "cen" rows covering every identity case.

    CEN is used because it declares both a code_col and a fuel_sub_col, so one fixture
    covers identity and subtype grouping. The rows cover: an EGE with a code reporting two
    subtypes, two EGEs with no code at all, a second EGE sharing a name under a different
    code, and a row with neither code nor name.

    Returns:
        pd.DataFrame: The CEN rows.
    """
    return pd.DataFrame(
        {
            "id_central": [1, 1, None, None, None, 2, None],
            "central": [
                "TER ALPHA",
                "TER ALPHA",
                "BESS BETA",
                "BESS BETA",
                "BESS GAMMA",
                "TER ALPHA",
                None,
            ],
            "tipo_tecnologia": ["Térmica"] * 2 + ["BESS"] * 3 + ["Térmica"] * 2,
            "subtipo_tecnologia": [
                "Diésel",
                "Fuel Oil",
                "Inyección",
                "Retiro",
                "Inyección",
                "Diésel",
                "Diésel",
            ],
        }
    )


@pytest.fixture
def cen_run(make_run: Callable[..., Run], cen_rows: pd.DataFrame) -> Run:
    """A "cen" run of the rows covering every identity case.

    Args:
        make_run (Callable[..., Run]): Factory for runs (s. conftest).
        cen_rows (pd.DataFrame): The CEN rows.

    Returns:
        Run: The CEN run.
    """
    return make_run("cen", cen_rows)


def _make_candidate() -> MatchCandidate:
    """Build a minimal GEM candidate for the fuzzy step to write into the df.

    Returns:
        MatchCandidate: One candidate with the fields `_write_candidate_into_df` needs.
    """
    return MatchCandidate(
        name="Plant A",
        primary_name="Plant A",
        norm_name="plant a",
        wt_string="plant:1.0",
        locator="gem",
        id="gem-1",
        fueltype="hydro",
        capacity=None,
        status=None,
        url=None,
        lat=1.0,
        lon=2.0,
        country="Brazil",
    )


class _DummyPipeline(BasePipeline):
    """Minimal concrete BasePipeline subclass for checking shared ALL_STEPS loop.

    Overrides `BasePipeline`'s dedupe/finalize steps so only the child STEPS-running mechanism
    itself is tested, not pipeline-specific behavior. BasePipeline's prep is used directly.
    """

    STEPS = ["_step_individual"]
    call_log: list[str] = []

    def _step_dedupe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.call_log.append("dedupe")
        return pd.DataFrame({"respondent-name": ["Plant A"]})  # eia's own name header

    def _step_individual(self, df: pd.DataFrame) -> pd.DataFrame:
        self.call_log.append("individual")
        return df

    def _step_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        self.call_log.append("finalize")
        return df


class _StopsEarlyPipeline(BasePipeline):
    """Concrete subclass using the real (inherited) `_step_dedupe`."""

    STEPS = ["_step_flag"]
    later_step_called: bool = False

    def _step_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        self.later_step_called = True
        return df


class _RealLoadPipeline(BasePipeline):
    """Concrete subclass adding no steps of its own, to exercise the real shared steps."""

    STEPS: list[str] = []


# ----------------------------------
# Tests
# ----------------------------------
class TestOperatorMetadata:
    """Tests for the invariants OPERATOR_METADATA has to satisfy."""

    def test_operator_pipelines(self) -> None:
        """Happy path for mappings, checking that only entsoe uses a non-"default" pipeline."""
        assert OPERATOR_METADATA["entsoe"].get("pipeline", "default") == "entsoe"
        assert OPERATOR_METADATA["eia"].get("pipeline", "default") == "default"
        assert OPERATOR_METADATA["adme"].get("pipeline", "default") == "default"

    def test_operator_columns_are_operatorinfo_keys(self) -> None:
        """Happy path: every OPERATOR_COLUMNS key must be a key of OperatorInfo.

        Each key is looked up on an operator's metadata to find its column. A key that doesn't
        exist on OperatorInfo finds nothing for every operator, silently dropping that column
        from matching and the output (as happened twice: "name_col" vs "entity_col", then
        "fuel_sub_col" vs "fuel_subtype_col").
        """
        assert set(OPERATOR_COLUMNS) <= set(OperatorInfo.__annotations__)

    def test_operators_with_fuel_sub_col_need_a_fuel_col(self) -> None:
        """Happy path: an operator declaring a fuel subtype declares its main fuel type too.

        The subtype only ever refines the main type (s. `_step_prepare_matching`), so one
        without the other is a metadata error - and one that would otherwise surface only when
        that operator happens to be run.
        """
        offenders = {
            op: meta["fuel_sub_col"]
            for op, meta in OPERATOR_METADATA.items()
            if meta.get("fuel_sub_col") and not meta.get("fuel_col")
        }
        assert not offenders

    def test_operators_have_a_name_col(self) -> None:
        """Failure path: Operators that can be name-matched require a name_col def.

        `BasePipeline.__init__` raises MissingDataError on an empty `name_col`, so an operator
        marked as name-matchable without one fails mid-run rather than at definition
        time. ADME is a structural exception: its raw CSVs have a MultiIndex (two-row)
        column header (s. its downloader's `header=[0, 1]`), which the pipeline's
        flat-header loading cannot read, so no single column name can be declared here.
        A new operator joining this set is a metadata error.
        """
        unready = {
            op
            for op, meta in OPERATOR_METADATA.items()
            if meta["name_matchable"] and not meta["name_col"]
        }
        assert unready == {"adme"}


class TestBasePipelineInit:
    """Tests for BasePipeline initialization."""

    def test_no_direct_instantiation(self, eia_run: Run) -> None:
        """Failure path to check that BasePipeline cannot be instantiated directly.

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        with pytest.raises(TypeError, match="BasePipeline must be subclassed"):
            BasePipeline(run=eia_run, output_dir=None, gem_loc=None, ppdb_loc=None)

    def test_unknown_operator(self, make_run: Callable[..., Run]) -> None:
        """Failure path: a run of an operator without metadata can't build a pipeline.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        with pytest.raises(ValueError, match="No metadata defined"):
            _RealLoadPipeline(
                run=make_run("not_a_real_operator"),
                output_dir=None,
                gem_loc=None,
                ppdb_loc=None,
            )

    def test_multi_country_operator_needs_a_known_zone(
        self, make_run: Callable[..., Run]
    ) -> None:
        """Failure path: a multi-country operator's run resolves its country by zone.

        Without one, ENTSO-E's EGEs would be matched against no country at all.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        with pytest.raises(ValueError, match="spans several countries"):
            _RealLoadPipeline(
                run=make_run("entsoe", zone=None),
                output_dir=None,
                gem_loc=None,
                ppdb_loc=None,
            )


class TestBasePipelineDedupe:
    """Tests for BasePipeline's shared dedupe step."""

    @staticmethod
    def _dedupe(run: Run) -> pd.DataFrame:
        """Run the real dedupe step on a run's rows.

        Args:
            run (Run): The operator's run.

        Returns:
            pd.DataFrame: The deduplicated frame, one row per unique EGE.
        """
        pipeline = _RealLoadPipeline(
            run=run, output_dir=None, gem_loc=None, ppdb_loc=None
        )
        return pipeline._step_dedupe(pd.DataFrame())

    def test_rows_without_code_use_name_as_id(self, cen_run: Run) -> None:
        """Happy path: rows with no code are identified by name, one EGE each.

        Deduping on the code column alone collapses every code-less row into a single EGE,
        because drop_duplicates treats all missing values as the same value. CEN leaves
        `id_central` empty for all 31 of its batteries, which published them as one row.

        Args:
            cen_run (Run): The synthetic CEN run.
        """
        df = self._dedupe(cen_run)

        assert sorted(df["central"]) == [
            "BESS BETA",
            "BESS GAMMA",
            "TER ALPHA",
            "TER ALPHA",
        ]

    def test_rows_without_code_or_name_are_dropped(self, cen_run: Run) -> None:
        """Happy path: a row with neither a code nor a name is dropped, not kept as an EGE.

        Such a row cannot be identified or matched, so keeping it invents an EGE that
        inherits whatever the first row of its group happened to hold.

        Args:
            cen_run (Run): The synthetic CEN run.
        """
        df = self._dedupe(cen_run)

        assert len(df) == 4
        assert df["central"].notna().all()

    def test_ege_with_diff_code_same_name_stay_separate(self, cen_run: Run) -> None:
        """Happy path: the code stays authoritative where a row has one.

        Names are not unique (3 ONS EGEs share one), so the name may only ever stand in
        for a missing code - never replace it.

        Args:
            cen_run (Run): The synthetic CEN run.
        """
        df = self._dedupe(cen_run)

        alpha = df[df["central"] == "TER ALPHA"]
        assert sorted(alpha["id_central"]) == [1.0, 2.0]

    def test_ege_fuel_subtypes_are_grouped(self, cen_run: Run) -> None:
        """Happy path: an EGE's combined fuel subtypes are its own, not its whole group's.

        The subtypes are combined over the same identity used for deduping, so a wrong
        identity leaks one EGE's fuels into another's (CEN's single code-less row reported
        "diesel oil, battery storage discharge injection, battery storage charge
        withdrawal", the union of every battery plus an unrelated diesel plant).

        Args:
            cen_run (Run): The synthetic CEN run.
        """
        df = self._dedupe(cen_run)
        subtypes = {
            name: set(str(sub).split(", "))
            for name, sub in zip(df["central"], df["subtipo_tecnologia"])
            if name.startswith("BESS")
        }

        # BETA reports two subtypes, GAMMA only the one of them that is its own
        assert len(subtypes["BESS BETA"]) == 2
        assert subtypes["BESS GAMMA"] < subtypes["BESS BETA"]

        # the same holds for the two coded EGEs sharing a name
        alpha = df[df["central"] == "TER ALPHA"].set_index("id_central")
        assert len(alpha.at[1.0, "subtipo_tecnologia"].split(", ")) == 2
        assert len(alpha.at[2.0, "subtipo_tecnologia"].split(", ")) == 1


class TestBasePipelineFuzzyMatch:
    """Tests for BasePipeline's shared fuzzy matching step."""

    def test_matcher_gets_the_pipelines_region_index(self, eia_run: Run) -> None:
        """Happy path: the name matcher checks regions with the pipeline's own index.

        If it built its own instead, every zone would read the admin-1 data again and
        ignore the run's `resources_dir` and `--update`.

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        region_reg = RegionRegistry()
        pipeline = _DummyPipeline(
            run=eia_run,
            output_dir=None,
            gem_loc=None,
            ppdb_loc=None,
            region_reg=region_reg,
        )

        with patch(f"{BASE_MODULE}.NameMatcher") as mock_matcher:
            pipeline._step_name_match(pd.DataFrame())

        assert mock_matcher.call_args.kwargs["region_reg"] is region_reg

    def test_matcher_gets_the_locators_candidate_frames(self, eia_run: Run) -> None:
        """Happy path: the pipeline hands each locator's rows to the matcher.

        The matcher takes frames rather than locator objects, so a locator whose frame
        is not passed on contributes no candidates at all -- and quietly, since None is
        a valid argument.

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        gem_frame = pd.DataFrame([{"plant_name": "Plant A"}])
        ppdb_frame = pd.DataFrame([{"Name": "Plant B"}])
        pipeline = _DummyPipeline(
            run=eia_run,
            output_dir=None,
            gem_loc=cast(GEMLocator, SimpleNamespace(df=gem_frame)),
            ppdb_loc=cast(OSMPPLocator, SimpleNamespace(df=ppdb_frame)),
        )

        with patch(f"{BASE_MODULE}.NameMatcher") as mock_matcher:
            pipeline._step_name_match(pd.DataFrame())

        kwargs = mock_matcher.call_args.kwargs
        assert kwargs["gem_df"] is gem_frame
        assert kwargs["ppdb_df"] is ppdb_frame

    @pytest.mark.parametrize("via", ["name_exact", "name_fuzzy"])
    def test_match_method_names_the_approach(self, eia_run: Run, via: str) -> None:
        """Happy path: the written match_method says how the name was matched.

        The value has to distinguish a key hit from token scoring, since a reviewer
        judges a match's trustworthiness by it (s. MATCH_METHOD_COLORS).

        Args:
            eia_run (Run): An (empty) EIA run.
            via (str): Parametrized approach the matcher reports.
        """
        pipeline = _DummyPipeline(
            run=eia_run, output_dir=None, gem_loc=None, ppdb_loc=None
        )
        df = pd.DataFrame({SYSOP_NAME_COL: ["Plant A"], SYSOP_FUEL_COL: ["hydro"]})
        candidate = _make_candidate()
        result = SimpleNamespace(
            matched=True,
            candidate=candidate,
            score=100.0,
            matched_via=via,
            to_dicts=lambda **kwargs: [],
        )

        with patch(f"{BASE_MODULE}.NameMatcher") as mock_matcher:
            mock_matcher.return_value.match.return_value = result
            out = pipeline._step_name_match(df)

        assert out.loc[0, "gem.match_method"] == f"gem_{via}"


class TestBasePipelineValidateRegion:
    """Tests for the region validation step's handling of an absent registry."""

    def test_no_registry_skips_the_step(self, eia_run: Run) -> None:
        """Happy path: without a region registry, no region validation is attempted.

        A run without a `resources_dir` has no admin-1 data to check against, and that
        must cost the operator its region columns rather than the whole run.

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        pipeline = _DummyPipeline(
            run=eia_run,
            output_dir=None,
            gem_loc=None,
            ppdb_loc=None,
            region_reg=None,
        )
        df = pd.DataFrame({SYSOP_REGION_COL: ["Sao Paulo"], "gem.lat": [1.0]})

        out = pipeline._step_validate_region(df)

        assert "region_match" not in out.columns
        assert "region_match_level" not in out.columns


class TestBasePipelineRunPipeline:
    """Tests for BasePipeline's run_pipeline method."""

    def test_executes_steps_in_order(self, eia_run: Run) -> None:
        """Happy path: run_pipeline calls ALL_STEPS entries, in order, exactly once.

        Verifies both the pipeline-specific STEPS and the automatic BasePipeline predefined
        dedupe & finalize steps. The real `_step_prepare_matching` runs in between "dedupe" and
        "individual" (publishing eia's own name header as "sysop.name", so that is checked).

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        pipeline = _DummyPipeline(
            run=eia_run, output_dir=None, gem_loc=None, ppdb_loc=None
        )
        df = pipeline.run_pipeline()

        assert pipeline.call_log == ["dedupe", "individual", "finalize"]
        assert list(df[SYSOP_NAME_COL]) == ["Plant A"]

    def test_stops_early_when_dedupe_is_empty(self, eia_run: Run):
        """Failure path: run_pipeline stops early when the first step returns empty.

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        pipeline = _StopsEarlyPipeline(
            run=eia_run, output_dir=None, gem_loc=None, ppdb_loc=None
        )
        df = pipeline.run_pipeline()

        assert df.empty
        assert pipeline.later_step_called is False

    def test_sysop_columns_are_generic(self, eia_run: Run) -> None:
        """Happy path: Relevant operator columns are renamed to generic "sysop.*".

        EIA names these "respondent-name"/"respondent"/"fueltype", but every operator
        publishes them under the same headers so runs stay comparable. Columns EIA doesn't
        have (e.g. fuel subtype, region) must produce no column at all, not an empty one.

        Args:
            eia_run (Run): An (empty) EIA run.
        """
        df_raw = pd.DataFrame(
            {"respondent-name": ["Plant A"], "respondent": ["P1"], "fueltype": ["NG"]}
        )
        pipeline = _DummyPipeline(
            run=eia_run, output_dir=None, gem_loc=None, ppdb_loc=None
        )
        df_prep = pipeline._step_prepare_matching(df_raw)

        # operators' headers are converted to generic "sysop.*" ones
        assert {SYSOP_NAME_COL, SYSOP_CODE_COL, SYSOP_FUEL_COL} <= set(df_prep.columns)
        assert not set(df_raw.columns) & set(df_prep.columns)

        # columns that the operator doesn't have configured don't exist in "sysop.*" form
        assert SYSOP_FUEL_SUB_COL not in df_prep.columns
        assert SYSOP_REGION_COL not in df_prep.columns


class TestBasePipelineOfficial:
    """Tests for coordinates an operator supplies itself ("official.*", e.g. a registry)."""

    @staticmethod
    def _pipeline(run: Run, region_reg: RegionRegistry | None = None) -> BasePipeline:
        """Build a pipeline running the real shared steps.

        Args:
            run (Run): The operator's run.
            region_reg (RegionRegistry | None): Region registry to validate with.

        Returns:
            BasePipeline: The pipeline.
        """
        return _RealLoadPipeline(
            run=run, output_dir=None, gem_loc=None, ppdb_loc=None, region_reg=region_reg
        )

    def test_dedupe_carries_official_columns(self, cen_run: Run) -> None:
        """Happy path: the caller's official columns survive the dedupe, per EGE.

        Dedupe keeps only the operator's own columns otherwise, which would silently
        drop the coordinates a processor added before handing over its rows.

        Args:
            cen_run (Run): The synthetic CEN run (codes 1 and 2 plus code-less rows).
        """
        official = cen_run.df["id_central"].map({1: -33.0, 2: -20.0})
        run = Run(
            sysop="cen",
            name="1h",
            zone=None,
            files=(),
            df=cen_run.df.assign(**{"official.lat": official}),
        )

        df = self._pipeline(run)._step_dedupe(pd.DataFrame())
        alpha = df[df["central"] == "TER ALPHA"].set_index("id_central")

        assert alpha.at[1.0, "official.lat"] == -33.0
        assert alpha.at[2.0, "official.lat"] == -20.0
        assert df.loc[df["id_central"].isna(), "official.lat"].isna().all()

    def test_official_region_stands_in_for_missing_region_col(
        self, make_run: Callable[..., Run]
    ) -> None:
        """Happy path: an official region is used where the raw data has none.

        CEN's generation data names no region, its registry does: without it, no name
        match of a CEN EGE could be checked against a region at all.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        df_raw = pd.DataFrame(
            {
                "central": ["TER ALPHA"],
                "official.region": ["Maule"],
                "official.lat": [-35.0],
            }
        )

        df = self._pipeline(make_run("cen"))._step_prepare_matching(df_raw)

        assert df.loc[0, SYSOP_REGION_COL] == "Maule"

    def test_operators_own_region_col_wins(self, make_run: Callable[..., Run]) -> None:
        """Happy path: an operator's own region column isn't replaced by an official one.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        df_raw = pd.DataFrame(
            {"nom_usina": ["UTE X"], "nom_estado": ["SP"], "official.region": ["RJ"]}
        )

        df = self._pipeline(make_run("ons"))._step_prepare_matching(df_raw)

        assert df.loc[0, SYSOP_REGION_COL] == "SP"

    def test_official_fueltype_is_the_eges(self, make_run: Callable[..., Run]) -> None:
        """Happy path: an official record carries its EGE's own fuel type.

        The record IS the EGE, so its fuel is the EGE's. A sibling inheriting the record
        then inherits a fuel to be validated against, as with every other locator.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        df_raw = pd.DataFrame(
            {
                "central": ["PFV A", "PE B"],
                "tipo_tecnologia": ["solar", "wind"],
                "official.lat": [-30.0, None],
            }
        )
        pipeline = self._pipeline(make_run("cen"))

        df = pipeline._step_prepare_matching(df_raw)

        assert df.loc[0, "official.fueltype"] == "solar"
        assert pd.isna(df.loc[1, "official.fueltype"])
        assert pipeline._matched_so_far == 1  # counted before matching, not by it

    def test_official_coordinates_win(self, make_run: Callable[..., Run]) -> None:
        """Happy path: an EGE's official coordinates win over any matched ones.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        df = pd.DataFrame(
            {
                "official.lat": [-30.0],
                "official.lon": [-70.0],
                "official.match_method": ["official_id_exact"],
                "gem.lat": [-31.0],
                "gem.lon": [-71.0],
                "gem.match_method": ["gem_name_fuzzy"],
            }
        )
        pipeline = self._pipeline(make_run("cen"))
        pipeline._create_match_method_columns(df)

        out = pipeline._step_finalize(df)

        assert out.loc[0, ["lat", "lon", "match_method"]].tolist() == [
            -30.0,
            -70.0,
            "official_id_exact",
        ]

    @staticmethod
    def _own_and_inherited() -> pd.DataFrame:
        """Two EGEs with official coordinates: one its own, one inherited from a sibling.

        Returns:
            pd.DataFrame: The working dataframe, after the sibling step.
        """
        return pd.DataFrame(
            {
                SYSOP_NAME_COL: ["PFV A", "BESS A"],
                SYSOP_FUEL_COL: ["solar", "battery storage"],
                SYSOP_REGION_COL: ["Maule", "Maule"],
                "official.lat": [-35.0, -35.0],
                "official.lon": [-71.0, -71.0],
                "official.match_method": ["official_id_exact", "official_sibling"],
                "official.fueltype": ["solar", "solar"],  # inherited with the record
                "sibling_of": [None, "PFV A"],
            }
        )

    def test_only_inherited_fueltype_is_validated(
        self, make_run: Callable[..., Run]
    ) -> None:
        """Happy path: only an EGE's own official record is exempt from the fuel check.

        An inherited record is an inference like any other sibling match: a battery taking
        a solar plant's coordinates is a fuel mismatch (CEN: "BESS PFV COYA" inherited
        "HP COYA", 1333 km away from its actual namesake "PFV COYA").

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        df = self._pipeline(make_run("cen"))._step_validate_fueltype(
            self._own_and_inherited()
        )

        assert df["fueltype_match_level"].tolist() == [None, "mismatch"]

    def test_only_inherited_region_is_validated(
        self, make_run: Callable[..., Run]
    ) -> None:
        """Happy path: only an EGE's own official record is exempt from the region check.

        Args:
            make_run (Callable[..., Run]): Factory for runs (s. conftest).
        """
        region_reg = cast(
            RegionRegistry, SimpleNamespace(classify_match=lambda *a, **k: "mismatch")
        )
        pipeline = self._pipeline(make_run("cen"), region_reg=region_reg)

        df = pipeline._step_validate_region(self._own_and_inherited())

        assert df["region_match_level"].tolist() == [None, "mismatch"]
