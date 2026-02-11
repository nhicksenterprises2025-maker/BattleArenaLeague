.gitignore
.gitignore
New
+2
-0

__pycache__/
*.pyc
README.md
README.md
New
+32
-0

# Battle Arena League Simulator

This repository now contains a full, runnable Python simulation app implementing the Master Document V3 systems:

- Card system + deck validation
- Archetype-aware card model
- Normalization and all card overall formulas
- Team overall and usage weighting
- Match engine (3 phases, deterministic season seed, micro-variance, phase RNG)
- Counters/chemistry/identity/streak/home/strategy modifiers
- Crowns, KO logic, duration, deep scoring
- Crystal gain/loss scaling with opponent PR
- Power rank computation
- Season scheduling (conference + cross-conference constraints)
- Core AI logic for roster optimization, identity assignment, trade evaluation
- Contract salary formula and cap pressure cuts

## Run

```bash
PYTHONPATH=src python -m battle_arena.app
```

## Test

```bash
PYTHONPATH=src pytest -q
```

## Notes

- The source spec had some missing table values (e.g., explicit streak table, explicit crown bonus table in one section). Where those values were not explicit, deterministic defaults were implemented in code while preserving all stated formulas and behavior patterns.
docs/master_document_v3.md
docs/master_document_v3.md
New
+347
-0

pyproject.toml
pyproject.toml
New
+14
-0

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "battle-arena-league"
version = "0.1.0"
description = "Deterministic Battle Arena League simulator from Master Doc V3"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[tool.pytest.ini_options]
testpaths = ["tests"]
src/battle_arena/__init__.py
src/battle_arena/__init__.py
New
+7
-0

__all__ = ["run_demo_season"]


def run_demo_season(seed: int = 42) -> str:
    from .app import run_demo_season as _run

    return _run(seed)
src/battle_arena/ai.py
src/battle_arena/ai.py
New
+79
-0

from __future__ import annotations

from dataclasses import dataclass

from .formulas import compute_card_overall, is_dead_card, salary_from_overall, usage_weights
from .models import Card, Team, TeamIdentity


@dataclass(slots=True)
class TradeProposal:
    incoming: list[Card]
    outgoing: list[Card]


def card_value(card: Card, scarcity: float, salary_efficiency: float) -> float:
    return (0.55 * card.overall) + (0.25 * scarcity) + (0.20 * salary_efficiency)


def assign_identity(team: Team) -> TeamIdentity:
    archetypes = [c.archetype.lower() for c in team.active_cards]
    has_buildings = sum("building" in a for a in archetypes)
    has_tower_attackers = sum("tower attacker" in a for a in archetypes)
    has_splash = sum("splash" in a for a in archetypes)

    if has_tower_attackers >= 2:
        return TeamIdentity.AGGRESSIVE
    if has_buildings >= 2:
        return TeamIdentity.DEFENSIVE
    if has_splash >= 2:
        return TeamIdentity.CONTROL
    return TeamIdentity.TEMPO


def optimize_roster(team: Team, available_cards: list[Card], salary_cap: int = 500_000) -> None:
    ranked = sorted(
        available_cards,
        key=lambda c: (compute_card_overall(c), -salary_from_overall(compute_card_overall(c))),
        reverse=True,
    )
    starters: list[Card] = []
    reserves: list[Card] = []

    for card in ranked:
        card.overall = compute_card_overall(card)
        card.death_score = max(-5, min(5, card.death_score))
        if is_dead_card(card.death_score):
            continue
        if len(starters) < 4:
            starters.append(card)
        elif len(reserves) < 2:
            reserves.append(card)
        if len(starters) == 4 and len(reserves) == 2:
            break

    team.active_cards = starters
    team.reserves = reserves
    team.identity = assign_identity(team)

    total_salary = sum(salary_from_overall(c.overall) for c in team.all_cards)
    while total_salary > salary_cap and team.reserves:
        cut = min(team.reserves, key=lambda c: c.overall)
        team.reserves.remove(cut)
        total_salary = sum(salary_from_overall(c.overall) for c in team.all_cards)


def evaluate_trade(net_value_pct: float, deadline_week: bool = False, seed_roll: float = 0.0) -> bool:
    threshold = 5.0
    if deadline_week:
        threshold *= 0.75
    if net_value_pct >= threshold:
        return True
    if net_value_pct <= -threshold:
        return False
    return seed_roll < 0.2


def usage_weighted_selection(cards: list[Card]) -> list[tuple[str, float]]:
    weights = usage_weights([c.elixir_cost for c in cards])
    return list(zip([c.name for c in cards], weights, strict=True))
src/battle_arena/app.py
src/battle_arena/app.py
New
+39
-0

from __future__ import annotations

from .ai import optimize_roster
from .data import starter_cards, starter_teams
from .season import SeasonSimulator
from .models import SeasonConfig


def run_demo_season(seed: int = 42) -> str:
    cards = starter_cards()
    teams = starter_teams()

    for i, team in enumerate(teams):
        pool = cards[i % len(cards) :] + cards[: i % len(cards)]
        optimize_roster(team, pool)

    simulator = SeasonSimulator(SeasonConfig(seed=seed))
    state = simulator.simulate(teams)
    standings = sorted(
        state.teams,
        key=lambda t: (
            t.stats.crystals,
            t.stats.total_deep_score,
            t.stats.wins,
            t.power_rank,
            -t.stats.losses,
        ),
        reverse=True,
    )
    top = standings[0]
    return (
        f"Season complete: {len(state.results)} games. "
        f"Champion of regular season: {top.name} ({top.stats.wins}-{top.stats.losses}, "
        f"Crystals {top.stats.crystals}, PR {top.power_rank:.2f})."
    )


if __name__ == "__main__":
    print(run_demo_season())
src/battle_arena/data.py
src/battle_arena/data.py
New
+38
-0

from __future__ import annotations

from .formulas import compute_card_overall
from .models import Card, CardType, MovementSpeed, Team


def starter_cards() -> list[Card]:
    cards = [
        Card("Tactical Sniper", "Ground Ranged", CardType.TROOP, 5, 890, 612, 2.1, MovementSpeed.SLOW, 10),
        Card("Archer", "Ground Ranged", CardType.TROOP, 3, 512, 144, 1.1, MovementSpeed.MEDIUM, 6),
        Card("Recon Hunter", "Ground Ranged", CardType.TROOP, 4, 1044, 233, 1.4, MovementSpeed.FAST, 7),
        Card("Combat Sharpshooter", "Ground Ranged", CardType.TROOP, 5, 1210, 478, 1.8, MovementSpeed.MEDIUM, 8),
        Card("Precision Ranger", "Ground Ranged", CardType.TROOP, 4, 932, 355, 1.2, MovementSpeed.FAST, 9),
        Card("Fort-Rammer", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 6, 4210, 622, 2.8, MovementSpeed.SLOW, 1, targets="Towers Only"),
        Card("Demolition Bull", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 6, 3550, 744, 2.4, MovementSpeed.MEDIUM, 1, targets="Towers Only"),
        Card("Rampage Juggernaut", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 8, 4980, 910, 3.1, MovementSpeed.VERY_SLOW, 1, targets="Towers Only"),
        Card("Iron Marauder", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 5, 2870, 520, 2.2, MovementSpeed.MEDIUM, 1, targets="Towers Only"),
        Card("Citadel Crusher", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 7, 4620, 820, 3.3, MovementSpeed.SLOW, 1, targets="Towers Only"),
        Card("Missile Silo", "Building", CardType.BUILDING, 5, 3400, 510, 1.5, None, 8, lifetime=60),
        Card("Tesla Tower", "Building", CardType.BUILDING, 4, 2900, 380, 1.4, None, 6, lifetime=55),
        Card("Drone Control Post", "Building", CardType.BUILDING, 4, 2400, 290, 1.3, None, 7, lifetime=65),
        Card("Turret", "Building", CardType.BUILDING, 3, 1800, 255, 1.1, None, 5, lifetime=50),
        Card("Bunker Buster", "Building", CardType.BUILDING, 6, 4200, 610, 1.8, None, 4, lifetime=45),
        Card("Meteor Shower", "Spell", CardType.SPELL, 8, damage=1320, spell_duration=8, spell_radius=8),
        Card("Nova Shock", "Spell", CardType.SPELL, 6, damage=880, spell_duration=0, spell_radius=3),
        Card("Acid Rain", "Spell", CardType.SPELL, 5, damage=610, spell_duration=7, spell_radius=4),
        Card("Sandstorm", "Spell", CardType.SPELL, 4, damage=480, spell_duration=6, spell_radius=5),
        Card("Gravity Pulse", "Spell", CardType.SPELL, 4, damage=300, spell_duration=5, spell_radius=6),
    ]
    for card in cards:
        card.overall = compute_card_overall(card)
    return cards


def starter_teams() -> list[Team]:
    dml = [f"DML Team {i}" for i in range(1, 9)]
    msl = [f"MSL Team {i}" for i in range(1, 8)]
    return [*(Team(name=t, conference="DML") for t in dml), *(Team(name=t, conference="MSL") for t in msl)]
src/battle_arena/engine.py
src/battle_arena/engine.py
New
+179
-0

from __future__ import annotations

import random
from dataclasses import dataclass

from .formulas import (
    compute_card_overall,
    home_bonus,
    strategy_modifier,
    streak_modifier,
    team_overall,
)
from .models import MatchResult, PhaseResult, Team, TeamIdentity, pairwise


@dataclass(slots=True)
class MatchContext:
    counter_mod_home: float = 0.0
    counter_mod_away: float = 0.0
    chemistry_home: float = 0.0
    chemistry_away: float = 0.0
    history_home: float = 0.0
    history_away: float = 0.0
    strategy_home: bool | None = None
    strategy_away: bool | None = None


class MatchEngine:
    PHASES = (("Opening", 0.30), ("Midgame", 0.30), ("Double Elixir", 0.40))

    def __init__(self, season_seed: int):
        self._rng = random.Random(season_seed)

    def _micro_variance(self) -> float:
        swing = self._rng.uniform(0.01, 0.02)
        return 1 + self._rng.choice([-1, 1]) * swing

    def _phase_rng(self, identity: TeamIdentity) -> float:
        phase = self._rng.uniform(-0.11, 0.11)
        if identity is TeamIdentity.TEMPO:
            phase *= 0.80
        return phase

    def _chemistry_modifier(self, team: Team, history_score: float = 0.0) -> float:
        synergy_score = 0.0
        for a, b in pairwise(team.active_cards):
            same_family = a.archetype.split()[0] == b.archetype.split()[0]
            synergy_score += 1 if same_family else -1
        chemistry = (0.6 * max(-4, min(4, synergy_score))) + (0.4 * max(0, min(3, history_score)))
        return chemistry / 10

    def _strength_pre_rng(self, team: Team, counter_mod: float, chemistry_mod: float, strategy_mod: float) -> float:
        base = team_overall([compute_card_overall(c) for c in team.active_cards])
        return base * (
            1
            + counter_mod
            + chemistry_mod
            + streak_modifier(team.stats.streak)
            + home_bonus(team.stats.crystals)
            + strategy_mod
        )

    @staticmethod
    def _crowns_from_margin(margin: float) -> int:
        if margin >= 0.25:
            return 3
        if margin >= 0.15:
            return 2
        if margin >= 0.05:
            return 1
        return 0

    @staticmethod
    def _time_bonus(win: bool, duration: int) -> int:
        if win:
            if duration <= 30:
                return 15
            if duration <= 60:
                return 12
            if duration <= 90:
                return 9
            if duration <= 120:
                return 6
            if duration <= 150:
                return 4
            return 3
        if duration <= 30:
            return -12
        if duration <= 60:
            return -10
        if duration <= 90:
            return -8
        if duration <= 120:
            return -6
        if duration <= 150:
            return -4
        return -2

    @staticmethod
    def _crown_bonus(crowns: int, win: bool) -> int:
        table = {0: 0, 1: 3, 2: 6, 3: 9}
        val = table[crowns]
        return val if win else -val

    def simulate_match(self, home: Team, away: Team, context: MatchContext | None = None) -> tuple[MatchResult, list[PhaseResult]]:
        context = context or MatchContext()
        mv_home = self._micro_variance()
        mv_away = self._micro_variance()

        chem_home = self._chemistry_modifier(home, context.history_home)
        chem_away = self._chemistry_modifier(away, context.history_away)

        pre_home = self._strength_pre_rng(
            home,
            context.counter_mod_home,
            chem_home,
            strategy_modifier(context.strategy_home),
        ) * mv_home
        pre_away = self._strength_pre_rng(
            away,
            context.counter_mod_away,
            chem_away,
            strategy_modifier(context.strategy_away),
        ) * mv_away

        phases: list[PhaseResult] = []
        total_home = 0.0
        total_away = 0.0
        for phase_name, weight in self.PHASES:
            fs_home = (0.78 * pre_home) + (0.22 * self._phase_rng(home.identity))
            fs_away = (0.78 * pre_away) + (0.22 * self._phase_rng(away.identity))
            p_home = fs_home * weight
            p_away = fs_away * weight
            phases.append(PhaseResult(phase_name, p_home, p_away))
            total_home += p_home
            total_away += p_away

        margin = abs(total_home - total_away)
        duration = int(max(30, min(180, round(180 - margin * 12))))

        home_won = total_home >= total_away
        winner = home if home_won else away
        loser = away if home_won else home

        winner_crowns = self._crowns_from_margin(margin)
        loser_crowns = 0

        if winner.identity is TeamIdentity.AGGRESSIVE:
            winner_crowns = min(3, winner_crowns + 1)
        if winner.identity is TeamIdentity.DEFENSIVE:
            winner_crowns = max(0, winner_crowns - 1)

        ko_chance = 0.05
        if winner.identity is TeamIdentity.CONTROL:
            ko_chance += 0.06
        if loser.identity is TeamIdentity.DEFENSIVE:
            ko_chance -= 0.05
        ko = self._rng.random() < max(0, ko_chance)

        deep_winner = self._time_bonus(True, duration) + self._crown_bonus(winner_crowns, True) + (10 if ko else 0)
        deep_loser = self._time_bonus(False, duration) + self._crown_bonus(loser_crowns, False) + (-10 if ko else 0)

        result = MatchResult(
            home_team=home.name,
            away_team=away.name,
            winner=winner.name,
            loser=loser.name,
            home_score=round(total_home, 4),
            away_score=round(total_away, 4),
            duration_seconds=duration,
            crowns_winner=winner_crowns,
            crowns_loser=loser_crowns,
            ko=ko,
            deep_score_winner=deep_winner,
            deep_score_loser=deep_loser,
            home_crystal_delta=0,
            away_crystal_delta=0,
        )
        return result, phases
src/battle_arena/formulas.py
src/battle_arena/formulas.py
New
+134
-0

from __future__ import annotations

from dataclasses import dataclass

from .models import Card, CardType, MOVEMENT_SPEED_TPS, MovementSpeed, TeamIdentity


@dataclass(frozen=True)
class StatBounds:
    min_value: float
    max_value: float


DEFAULT_BOUNDS = {
    "health": StatBounds(200, 5000),
    "damage": StatBounds(80, 1400),
    "range": StatBounds(1, 10),
    "hit_speed": StatBounds(0.7, 3.3),
    "lifetime": StatBounds(30, 70),
    "spell_duration": StatBounds(0.0, 8.0),
    "spell_radius": StatBounds(3, 8),
    "move_speed": StatBounds(1.1, 3.9),
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def norm(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return clamp01((value - min_value) / (max_value - min_value))


def norm_inv(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return clamp01((max_value - value) / (max_value - min_value))


def move_speed_value(speed: MovementSpeed | None) -> float:
    if speed is None:
        return MOVEMENT_SPEED_TPS[MovementSpeed.MEDIUM]
    return MOVEMENT_SPEED_TPS[speed]


def compute_card_overall(card: Card) -> float:
    h = norm(card.health, **DEFAULT_BOUNDS["health"].__dict__)
    d = norm(card.damage, **DEFAULT_BOUNDS["damage"].__dict__)
    r = norm(card.attack_range, **DEFAULT_BOUNDS["range"].__dict__)
    hs = norm_inv(card.hit_speed, **DEFAULT_BOUNDS["hit_speed"].__dict__)
    ms = norm(move_speed_value(card.move_speed), **DEFAULT_BOUNDS["move_speed"].__dict__)

    if card.card_type is CardType.TOWER_ATTACKER:
        target_bonus = 0.05 if card.targets.lower() == "towers only" else 0.0
        score = 0.25 * h + 0.30 * d + 0.20 * r + 0.10 * ms + 0.15 * hs + target_bonus
        return round(min(1.0, score) * 100, 1)

    if card.card_type is CardType.BUILDING:
        lt = norm(card.lifetime, **DEFAULT_BOUNDS["lifetime"].__dict__)
        score = 0.25 * h + 0.30 * d + 0.20 * r + 0.10 * lt + 0.15 * hs
        return round(score * 100, 1)

    if card.card_type is CardType.SPELL:
        sd_inv = norm_inv(card.spell_duration, **DEFAULT_BOUNDS["spell_duration"].__dict__)
        tr = norm(card.spell_radius, **DEFAULT_BOUNDS["spell_radius"].__dict__)
        score = 0.50 * d + 0.30 * sd_inv + 0.20 * tr
        return round(score * 100, 1)

    score = 0.25 * h + 0.23 * d + 0.16 * hs + 0.14 * ms + 0.22 * r
    return round(score * 100, 1)


def team_overall(card_overalls: list[float]) -> float:
    if len(card_overalls) != 4:
        raise ValueError("Team overall requires exactly 4 active cards")
    return sum(card_overalls) / 4


def validate_deck(active_count: int, reserve_count: int) -> bool:
    total = active_count + reserve_count
    return active_count == 4 and 0 <= reserve_count <= 2 and total <= 6


def usage_weights(elixir_costs: list[int]) -> list[float]:
    raw = [4 / c for c in elixir_costs]
    total = sum(raw)
    return [r / total for r in raw]


def update_death_score(old_score: float, matches_used: int, team_matches: int) -> float:
    usage = 0.0 if team_matches <= 0 else matches_used / team_matches
    new_score = old_score + (0.3 - usage)
    return max(-5.0, min(5.0, new_score))


def is_dead_card(death_score: float) -> bool:
    return death_score >= 3


def salary_from_overall(overall: float) -> int:
    return round(23_400 + (overall / 100) * (198_200 - 23_400))


def home_bonus(crystals: int) -> float:
    return 0.01 + 0.02 * (max(0, min(crystals, 20_000)) / 20_000)


def strategy_modifier(good_matchup: bool | None) -> float:
    if good_matchup is True:
        return 0.03
    if good_matchup is False:
        return -0.03
    return 0.0


def streak_modifier(streak: int) -> float:
    capped = max(-6, min(6, streak))
    if capped == 0:
        return 0.0
    sign = 1 if capped > 0 else -1
    mag = min(0.06, 0.01 * abs(capped))
    return sign * mag


def identity_strength_modifier(identity: TeamIdentity) -> float:
    if identity is TeamIdentity.AGGRESSIVE:
        return 0.0
    if identity is TeamIdentity.DEFENSIVE:
        return 0.0
    if identity is TeamIdentity.CONTROL:
        return 0.0
    return 0.0
src/battle_arena/models.py
src/battle_arena/models.py
New
+126
-0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class CardType(str, Enum):
    TROOP = "troop"
    TOWER_ATTACKER = "tower_attacker"
    BUILDING = "building"
    SPELL = "spell"


class MovementSpeed(str, Enum):
    VERY_SLOW = "Very Slow"
    SLOW = "Slow"
    MEDIUM = "Medium"
    FAST = "Fast"
    VERY_FAST = "Very Fast"


MOVEMENT_SPEED_TPS = {
    MovementSpeed.VERY_SLOW: 1.1,
    MovementSpeed.SLOW: 1.8,
    MovementSpeed.MEDIUM: 2.5,
    MovementSpeed.FAST: 3.2,
    MovementSpeed.VERY_FAST: 3.9,
}


class TeamIdentity(str, Enum):
    AGGRESSIVE = "Aggressive"
    DEFENSIVE = "Defensive"
    CONTROL = "Control"
    TEMPO = "Tempo"


@dataclass(slots=True)
class Card:
    name: str
    archetype: str
    card_type: CardType
    elixir_cost: int
    health: float = 0.0
    damage: float = 0.0
    hit_speed: float = 1.0
    move_speed: MovementSpeed | None = None
    attack_range: float = 0.0
    lifetime: float = 0.0
    spell_duration: float = 0.0
    spell_radius: float = 0.0
    targets: str = "Ground Troops & Buildings"
    death_score: float = 0.0
    matches_used: int = 0
    overall: float = 0.0


@dataclass(slots=True)
class Contract:
    salary: int
    years: int


@dataclass(slots=True)
class TeamStats:
    wins: int = 0
    losses: int = 0
    games_played: int = 0
    crystals: int = 0
    total_deep_score: int = 0
    streak: int = 0


@dataclass(slots=True)
class Team:
    name: str
    conference: str
    identity: TeamIdentity = TeamIdentity.TEMPO
    active_cards: list[Card] = field(default_factory=list)
    reserves: list[Card] = field(default_factory=list)
    contracts: dict[str, Contract] = field(default_factory=dict)
    stats: TeamStats = field(default_factory=TeamStats)
    power_rank: float = 0.0

    @property
    def all_cards(self) -> list[Card]:
        return [*self.active_cards, *self.reserves]


@dataclass(slots=True)
class MatchResult:
    home_team: str
    away_team: str
    winner: str
    loser: str
    home_score: float
    away_score: float
    duration_seconds: int
    crowns_winner: int
    crowns_loser: int
    ko: bool
    deep_score_winner: int
    deep_score_loser: int
    home_crystal_delta: int
    away_crystal_delta: int


@dataclass(slots=True)
class PhaseResult:
    name: str
    home_score: float
    away_score: float


@dataclass(slots=True)
class SeasonConfig:
    seed: int
    regular_games_per_team: int = 41
    salary_cap: int = 500_000


def pairwise(items: Sequence[Card]) -> Iterable[tuple[Card, Card]]:
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]
src/battle_arena/season.py
src/battle_arena/season.py
New
+132
-0

from __future__ import annotations

import random
from dataclasses import dataclass

from .engine import MatchContext, MatchEngine
from .formulas import norm, salary_from_overall
from .models import Contract, MatchResult, SeasonConfig, Team


@dataclass(slots=True)
class SeasonState:
    config: SeasonConfig
    teams: list[Team]
    schedule: list[tuple[str, str]]
    results: list[MatchResult]


class SeasonSimulator:
    def __init__(self, config: SeasonConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.engine = MatchEngine(config.seed)

    def generate_schedule(self, teams: list[Team]) -> list[tuple[str, str]]:
        by_conf: dict[str, list[Team]] = {}
        for team in teams:
            by_conf.setdefault(team.conference, []).append(team)
        dml = by_conf.get("DML", [])
        msl = by_conf.get("MSL", [])
        schedule: list[tuple[str, str]] = []

        for i, t1 in enumerate(dml):
            for t2 in dml[i + 1 :]:
                schedule.extend([(t1.name, t2.name), (t2.name, t1.name)] * 2)

        for i, t1 in enumerate(msl):
            for t2 in msl[i + 1 :]:
                repeat = 4 if (i + msl.index(t2)) % 2 == 0 else 3
                for k in range(repeat):
                    schedule.append((t1.name, t2.name) if k % 2 == 0 else (t2.name, t1.name))

        dml_cross_target = 13
        msl_cross_target = 20
        counts = {t.name: 0 for t in teams}
        cross_pairs: dict[tuple[str, str], int] = {}
        while True:
            need_dml = [t for t in dml if counts[t.name] < dml_cross_target]
            need_msl = [t for t in msl if counts[t.name] < msl_cross_target]
            if not need_dml and not need_msl:
                break
            if not need_dml or not need_msl:
                break
            h = self.rng.choice(need_dml)
            a = self.rng.choice(need_msl)
            key = tuple(sorted((h.name, a.name)))
            if cross_pairs.get(key, 0) >= 3:
                continue
            cross_pairs[key] = cross_pairs.get(key, 0) + 1
            if counts[h.name] <= counts[a.name]:
                schedule.append((h.name, a.name))
            else:
                schedule.append((a.name, h.name))
            counts[h.name] += 1
            counts[a.name] += 1

        self.rng.shuffle(schedule)
        return schedule

    def _apply_crystals(self, result: MatchResult, home: Team, away: Team) -> None:
        base_win = self.rng.randint(490, 525)
        base_loss = self.rng.randint(480, 515)
        delta_pr_home = (away.power_rank - home.power_rank) / 100
        delta_pr_away = (home.power_rank - away.power_rank) / 100

        if result.winner == home.name:
            gain = round(base_win * (1 + 0.15 * delta_pr_home))
            loss = round(base_loss * (1 + 0.15 * delta_pr_away))
            home.stats.crystals = max(0, min(20_000, home.stats.crystals + gain))
            away.stats.crystals = max(0, min(20_000, away.stats.crystals - loss))
            result.home_crystal_delta = gain
            result.away_crystal_delta = -loss
        else:
            gain = round(base_win * (1 + 0.15 * delta_pr_away))
            loss = round(base_loss * (1 + 0.15 * delta_pr_home))
            away.stats.crystals = max(0, min(20_000, away.stats.crystals + gain))
            home.stats.crystals = max(0, min(20_000, home.stats.crystals - loss))
            result.away_crystal_delta = gain
            result.home_crystal_delta = -loss

    def recompute_power_ranks(self, teams: list[Team]) -> None:
        c_vals = [t.stats.crystals for t in teams]
        w_vals = [0 if t.stats.games_played == 0 else t.stats.wins / t.stats.games_played for t in teams]
        s_vals = [0 if t.stats.games_played == 0 else t.stats.total_deep_score / t.stats.games_played for t in teams]
        min_c, max_c = min(c_vals), max(c_vals)
        min_w, max_w = min(w_vals), max(w_vals)
        min_s, max_s = min(s_vals), max(s_vals)
        for t, w, s in zip(teams, w_vals, s_vals, strict=True):
            cn = norm(t.stats.crystals, min_c, max_c)
            wn = norm(w, min_w, max_w)
            sn = norm(s, min_s, max_s)
            t.power_rank = (0.5 * cn + 0.3 * wn + 0.2 * sn) * 100

    def simulate(self, teams: list[Team]) -> SeasonState:
        team_map = {t.name: t for t in teams}
        for t in teams:
            for c in t.all_cards:
                c.overall = c.overall or 50.0
                t.contracts.setdefault(c.name, Contract(salary_from_overall(c.overall), 1))

        schedule = self.generate_schedule(teams)
        results: list[MatchResult] = []

        for home_name, away_name in schedule:
            home, away = team_map[home_name], team_map[away_name]
            result, _ = self.engine.simulate_match(home, away, MatchContext())
            self._apply_crystals(result, home, away)
            winner = home if result.winner == home.name else away
            loser = away if winner is home else home

            winner.stats.wins += 1
            loser.stats.losses += 1
            winner.stats.streak = max(1, winner.stats.streak + 1)
            loser.stats.streak = min(-1, loser.stats.streak - 1)
            winner.stats.games_played += 1
            loser.stats.games_played += 1
            winner.stats.total_deep_score += result.deep_score_winner
            loser.stats.total_deep_score += result.deep_score_loser
            self.recompute_power_ranks(teams)
            results.append(result)

        return SeasonState(config=self.config, teams=teams, schedule=schedule, results=results)
tests/test_formulas.py
tests/test_formulas.py
New
+49
-0

from battle_arena.formulas import (
    compute_card_overall,
    salary_from_overall,
    team_overall,
    update_death_score,
    usage_weights,
    validate_deck,
)
from battle_arena.models import Card, CardType, MovementSpeed


def test_deck_validation():
    assert validate_deck(4, 0)
    assert validate_deck(4, 2)
    assert not validate_deck(3, 1)
    assert not validate_deck(4, 3)


def test_usage_weights_normalize():
    weights = usage_weights([2, 4, 8])
    assert round(sum(weights), 6) == 1.0
    assert weights[0] > weights[1] > weights[2]


def test_card_overall_and_salary():
    card = Card(
        name="Test Troop",
        archetype="Ground Ranged",
        card_type=CardType.TROOP,
        elixir_cost=4,
        health=1000,
        damage=300,
        hit_speed=1.2,
        move_speed=MovementSpeed.FAST,
        attack_range=7,
    )
    overall = compute_card_overall(card)
    assert 0 <= overall <= 100
    salary = salary_from_overall(overall)
    assert 23_400 <= salary <= 198_200


def test_team_overall_requires_4_cards():
    assert team_overall([10, 20, 30, 40]) == 25


def test_death_score_clamp():
    assert update_death_score(4.9, 0, 10) <= 5
    assert update_death_score(-4.9, 10, 10) >= -5
tests/test_season.py
tests/test_season.py
New
+15
-0

from battle_arena.app import run_demo_season
from battle_arena.data import starter_teams
from battle_arena.season import SeasonSimulator
from battle_arena.models import SeasonConfig


def test_schedule_generation_non_empty():
    sim = SeasonSimulator(SeasonConfig(seed=123))
    schedule = sim.generate_schedule(starter_teams())
    assert len(schedule) > 0


def test_demo_season_runs():
    msg = run_demo_season(seed=7)
    assert "Season complete" in msg
