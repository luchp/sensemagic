"""Driver-database helpers for the web page: load, filter, list, rank.

Thin wrapper around the ``audioshape`` package (pure physics/ranking,
installed from https://github.com/luchp/audioshape). No plotting here --
Plotly figures live in the router, mirroring ``app_rectifier.py``'s
convention of keeping page logic separate from figure-building code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from audioshape.database import ParseResult, parse_database
from audioshape.driver import Driver
from audioshape.ranking import Evaluation, evaluate, rank
from audioshape.scenario import Scenario

DEFAULT_DB_PATH = Path(__file__).parent / "data" / "VituixCAD_driver_db.txt"

EMPTY_VALUE = ""  # combobox value for "<empty>"
EMPTY_LABEL = "\u2014 empty \u2014"

# role -> (band_low, band_high, doppler_ref) Scenario attribute names
ROLE_BANDS = {
    "sub": ("f_low", "f_split", "f_split"),
    "attack": ("f_split", "f_high", "f_high"),
}


@lru_cache(maxsize=1)
def load_drivers(path: str | Path = DEFAULT_DB_PATH) -> ParseResult:
    """Parse the bundled VituixCAD database once and cache the result."""
    return parse_database(path)


def band_for_role(scenario: Scenario, role: str) -> tuple[float, float, float]:
    lo_name, hi_name, ref_name = ROLE_BANDS[role]
    return (getattr(scenario, lo_name), getattr(scenario, hi_name),
            getattr(scenario, ref_name))


def filter_by_size(drivers: list[Driver], size_min: float,
                   size_max: float) -> list[Driver]:
    return [d for d in drivers if size_min <= d.size_in <= size_max]


@dataclass(frozen=True)
class Option:
    """One combobox entry."""
    value: str   # driver.label(), or "" for <empty>
    text: str    # what the user sees


def alphabetical_options(drivers: list[Driver]) -> list[Option]:
    """`<empty>` first, then every driver sorted by label."""
    ordered = sorted(drivers, key=lambda d: d.label().lower())
    opts = [Option(EMPTY_VALUE, EMPTY_LABEL)]
    opts += [Option(d.label(), f"{d.label()} ({d.size_in:g}\")") for d in ordered]
    return opts


def ranked_options(drivers: list[Driver], scenario: Scenario, role: str,
                   n_units: int, size_min: float,
                   size_max: float) -> tuple[list[Option], list[Evaluation]]:
    """Rank the size-filtered drivers for `role` and format as options.

    Returns (options, evaluations) -- evaluations are in the same (ranked)
    order as the options that follow the leading `<empty>` entry.
    """
    band_low, band_high, doppler_ref = band_for_role(scenario, role)
    evals = rank(drivers, scenario, n_units=n_units,
                min_size_in=size_min, max_size_in=size_max,
                band_low=band_low, band_high=band_high, doppler_ref=doppler_ref)
    opts = [Option(EMPTY_VALUE, EMPTY_LABEL)]
    for i, ev in enumerate(evals, 1):
        flag = "" if ev.feasible else " [!]"
        opts.append(Option(
            ev.driver.label(),
            f"#{i} {ev.driver.label()} \u2014 {100*ev.total_distortion:.1f}% dist{flag}"))
    return opts, evals


def find_driver(drivers: list[Driver], label: str) -> Driver | None:
    """Exact label match (as produced by `Driver.label()` / combobox values)."""
    if not label:
        return None
    for d in drivers:
        if d.label() == label:
            return d
    return None


def evaluate_role(driver: Driver, scenario: Scenario, role: str,
                  n_units: int) -> Evaluation:
    band_low, band_high, doppler_ref = band_for_role(scenario, role)
    return evaluate(driver, scenario, n_units=n_units,
                    band_low=band_low, band_high=band_high,
                    doppler_ref=doppler_ref)
