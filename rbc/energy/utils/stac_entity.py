"""ENTITY TYPES.

Operator-neutral rules for the STAC energy extension's entity types (energy:entity_type,
energy:facility_name), applied to an operator's EGEs before they become STAC Items.
"""

import re
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd

from rbc.coordinates.mappings import OPERATOR_METADATA
from rbc.coordinates.utils.tokenizer import NORM_GENERIC_UNIT_TOKENS, NameTokenizer
from rbc.coordinates.utils.values import normalize_name

EARTH_RADIUS_KM = 6371.0
UNIT_TOKENS = frozenset(
    NORM_GENERIC_UNIT_TOKENS
)  # e.g. "bloco" = never in facility_name


class EntityTypeDefiner:
    """For STAC, defines a sysop's EGEs' entity types using names and codes.

    Attributes:
        sysop (str): The energy system operator, e.g. "ons".
        max_km (float): How far apart parts of a facility linked by name may lie.
        tok (NameTokenizer): The operator's tokenizer, as its coordinate finding has it.
    """

    def __init__(self, sysop: str, max_km: float = 1.0) -> None:
        """Initialize the definer with the operator's name vocabulary.

        Args:
            sysop (str): The operator, e.g. "ons".
            max_km (float, optional): How far apart parts linked by name may lie.
                Defaults to 1.0 km.

        Raises:
            ValueError: If the operator's vocabulary is per country (e.g. ENTSO-E's),
                which isn't supported yet.
        """
        self.sysop = sysop
        self.max_km = max_km

        meta = OPERATOR_METADATA[sysop]
        if meta["country"] == "Europe":  # a vocabulary per country (s. BasePipeline)
            raise ValueError(
                f"'{sysop}' has a name vocabulary per country: not supported."
            )

        names = cast(dict[str, str], meta.get("name_mapping", {}))
        self.tok = NameTokenizer(name_mapping={**meta.get("fuel_mapping", {}), **names})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_groups(
        self, df_ege: pd.DataFrame, active_periods: pd.Series, totals: pd.Series
    ) -> pd.DataFrame:
        """Find the EGEs that are parts ('group') of one facility, and name the facility.

        Two EGEs are parts of one facility if both report in a common period, under
        different names (an identical name is the same EGE listed again, not a part;
        roman unit numerals count as numbers), and
        - share a project code (e.g. ANEEL's CEG), or
        - share their name's discriminator words and lie within `max_km`.
        A facility joins all EGEs linked to each other, also through others (A-B, B-C).
        Its name runs from the first to the last discriminator word all its parts share
        (s. `_name_facility`), else, if linked by a code, it is the name of the part
        that generated most. A facility without a name leaves its EGEs as they are.

        Args:
            df_ege (pd.DataFrame): The candidate EGEs (e.g. single plants, not plant groups),
                with "id", "entity_name", "code" (missing if none), "lat" and "lon".
            active_periods (pd.Series): The periods each EGE reports values in (a set,
                e.g. of months), indexed by ID.
            totals (pd.Series): Each EGE's total generation, indexed by ID.

        Returns:
            pd.DataFrame: Indexed by the parts' IDs, with "entity_type" ("group") and
                "facility_name"; EGEs that aren't a part are left out.
        """
        df_ege = df_ege.set_index("id")
        names = df_ege["entity_name"].where(
            df_ege["entity_name"].map(lambda n: isinstance(n, str))
        )
        df_ege = df_ege.assign(
            key=names.map(self.tok.get_discriminator_key, na_action="ignore"),
            tokens=names.map(self.tok.get_canonical_tokens, na_action="ignore"),
            active_periods=[active_periods.get(i, set()) for i in df_ege.index],
            total=totals.reindex(df_ege.index).fillna(0.0),
        )

        parent = {
            i: i for i in df_ege.index
        }  # union-find: each EGE's link to its facility
        by_code: set[str] = set()  # EGEs linked by a code
        for col in ("code", "key"):
            for _, members in df_ege.dropna(subset=[col]).groupby(col):
                for a, b in combinations(members.index, 2):
                    if self._are_parts(
                        df_ege.loc[a], df_ege.loc[b], by_name=col == "key"
                    ):
                        parent[self._find_root(parent, a)] = self._find_root(parent, b)
                        if col == "code":
                            by_code.update((a, b))

        facilities = pd.Series(
            [self._find_root(parent, i) for i in df_ege.index], index=df_ege.index
        )
        rows = {}
        for _, members in df_ege.groupby(facilities):
            if len(members) < 2:
                continue
            linked_by_code = bool(by_code & set(members.index))
            if name := self._name_facility(members, linked_by_code):
                rows.update({i: ("group", name) for i in members.index})
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=["entity_type", "facility_name"]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_root(parent: dict[str, str], i: str) -> str:
        """Find the facility an EGE belongs to, as the EGE all its links lead to.

        Shortens the links it follows, so later lookups take fewer steps.

        Args:
            parent (dict[str, str]): Each EGE's link to another of its facility.
            i (str): The EGE's ID.

        Returns:
            str: The ID of the EGE that stands for its facility.
        """
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _are_parts(self, a: pd.Series, b: pd.Series, by_name: bool) -> bool:
        """Whether two EGEs that share a code or name key are parts of one facility.

        Args:
            a (pd.Series): The one EGE (s. `find_groups`).
            b (pd.Series): The other EGE.
            by_name (bool): Whether they share a name key (proximity check is necessary)
                or not (share a code; proximity check is NOT necessary).

        Returns:
            bool: True if both report in a common period, under different names, and
                (if linked by name) lie within `max_km`.
        """
        if not a["active_periods"] & b["active_periods"]:
            return False  # one after the other: the same EGE renamed, not both subparts

        if a["tokens"] == b["tokens"]:
            return False  # the same name: the same EGE listed twice, not both subparts

        if not by_name:
            return True

        located = a[["lat", "lon"]].notna().all() and b[["lat", "lon"]].notna().all()
        return bool(located and self._get_distance_km(a, b) <= self.max_km)

    def _name_facility(self, members: pd.DataFrame, linked_by_code: bool) -> str | None:
        """Name a facility after the discriminator words its parts share, else its largest.

        Args:
            members (pd.DataFrame): The facility's parts (s. `find_groups`).
            linked_by_code (bool): Whether any of them are linked by a code.

        Returns:
            str | None: The largest part's name from the first to the last discriminator
                word all parts share (e.g. "Porto do Pecém" for "Porto do Pecém I" and
                "II"), widened by adjacent energy words all parts share (e.g. "Fontes
                Solar", but not "Santana Bloco"); else the largest part's name if linked
                by a code; else None.
        """
        largest = members["entity_name"][members["total"].idxmax()]
        keys = members["key"]
        shared = (
            set.intersection(*(set(k.split()) for k in keys))
            if keys.notna().all()
            else set()
        )
        words = re.split(r"[\s_]+", largest.strip())  # e.g. "EC_TEC_NIKOLA_TESLA_B1"
        hits = [
            i
            for i, w in enumerate(words)
            if set((normalize_name(w) or "").split()) & shared
        ]
        if not hits:
            return largest if linked_by_code else None

        every = [
            {normalize_name(w) for w in re.split(r"[\s_]+", n.strip())}
            for n in members["entity_name"]
        ]
        start, end = hits[0], hits[-1]
        while start > 0 and self._extends_name(words[start - 1], every):
            start -= 1
        while end < len(words) - 1 and self._extends_name(words[end + 1], every):
            end += 1
        return " ".join(words[start : end + 1])

    def _extends_name(self, word: str, every: list[set[str]]) -> bool:
        """Whether a word next to a facility's name joins it: an energy word all parts have.

        Args:
            word (str): The word before or after the name's words so far.
            every (list[set[str]]): Each part's name, as a set of normalized words.

        Returns:
            bool: True if it is an energy word and in every part's name.
        """
        return self._is_energy_word(word) and all(
            normalize_name(word) in words for words in every
        )

    def _is_energy_word(self, word: str) -> bool:
        """Whether a word is a generic energy word ("Solar", "Usina"), not a unit word.

        Args:
            word (str): One word of a name.

        Returns:
            bool: True if all its tokens are generic and none is a unit word ("Bloco").
        """
        wt = self.tok.weighted_tokenize(word)
        return (
            bool(wt.tokens)
            and all(kind == "generic" for kind in wt.types)
            and not set(wt.tokens) & UNIT_TOKENS
        )

    @staticmethod
    def _get_distance_km(a: pd.Series, b: pd.Series) -> float:
        """Give the great-circle distance between two EGEs' points in km.

        Args:
            a (pd.Series): One EGE, with "lat" and "lon".
            b (pd.Series): The other EGE.

        Returns:
            float: Their distance in km.
        """
        lat1, lon1, lat2, lon2 = np.radians([a["lat"], a["lon"], b["lat"], b["lon"]])
        h = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))
