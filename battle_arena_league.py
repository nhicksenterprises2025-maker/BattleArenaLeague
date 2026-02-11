from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# Constants from Master Doc
# =========================
MIN_SALARY = 23_400
MAX_SALARY = 198_200
SALARY_CAP = 500_000
MAX_CRYSTALS = 20_000

MOVEMENT_SPEED_TPS: Dict[str, float] = {
    "Very Slow": 1.1,
    "Slow": 1.8,
    "Medium": 2.5,
    "Fast": 3.2,
    "Very Fast": 3.9,
}

STREAK_TABLE = {
    -5: -0.06,
    -4: -0.05,
    -3: -0.04,
    -2: -0.03,
    -1: -0.015,
    0: 0.0,
    1: 0.015,
    2: 0.03,
    3: 0.04,
    4: 0.05,
    5: 0.06,
}

CROWN_BONUS = {0: 0, 1: 2, 2: 4, 3: 7}
WIN_TIME_BONUS = [(30, 15), (60, 12), (90, 9), (120, 6), (150, 4), (180, 3)]
LOSS_TIME_BONUS = [(30, -12), (60, -10), (90, -8), (120, -6), (150, -4), (180, -2)]


class CardType(str, Enum):
    TROOP = "Troop"
    TOWER_ATTACKER = "Tower Attacker"
    BUILDING = "Building"
    SPELL = "Spell"


class TeamIdentity(str, Enum):
    AGGRESSIVE = "Aggressive"
    DEFENSIVE = "Defensive"
    CONTROL = "Control"
    TEMPO = "Tempo"


@dataclass
class Card:
    name: str
    archetype: str
    card_type: CardType
    health: float = 0.0
    damage: float = 0.0
    hit_speed: float = 1.0
    move_speed: str = "Medium"
    attack_range: float = 1.0
    lifetime: float = 0.0
    spell_duration: float = 0.0
    tile_radius: float = 0.0
    elixir_cost: int = 4
    targets: str = "Ground Troops & Buildings"
    spawn_count: int = 1
    splash_radius: float = 0.0
    matches_used: int = 0
    death_score: float = 0.0
    seasons_together: int = 0


@dataclass
class Contract:
    salary: int
    years: int


@dataclass
class TeamCard:
    card: Card
    contract: Contract


@dataclass
class Deck:
    active: List[TeamCard]
    reserves: List[TeamCard] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.active) != 4:
            raise ValueError("Deck must have exactly 4 active cards")
        if not 0 <= len(self.reserves) <= 2:
            raise ValueError("Deck must have 0-2 reserves")
        if len(self.active) + len(self.reserves) > 6:
            raise ValueError("Deck size cannot exceed 6")

    @property
    def all_cards(self) -> List[TeamCard]:
        return [*self.active, *self.reserves]


@dataclass
class Team:
    name: str
    conference: str
    deck: Deck
    identity: TeamIdentity = TeamIdentity.TEMPO
    wins: int = 0
    losses: int = 0
    crystals: int = 0
    total_deep_score: int = 0
    streak: int = 0
    power_rank: float = 50.0

    @property
    def games_played(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct(self) -> float:
        return self.wins / self.games_played if self.games_played else 0.0

    @property
    def avg_deep_score(self) -> float:
        return self.total_deep_score / self.games_played if self.games_played else 0.0

    @property
    def cap_hit(self) -> int:
        return sum(tc.contract.salary for tc in self.deck.all_cards)


# ===============
# Math helpers
# ===============
def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def linear_norm(x: float, x_min: float, x_max: float) -> float:
    if x_max == x_min:
        return 0.0
    return clamp((x - x_min) / (x_max - x_min), 0.0, 1.0)


def inverse_norm(x: float, x_min: float, x_max: float) -> float:
    if x_max == x_min:
        return 0.0
    return clamp((x_max - x) / (x_max - x_min), 0.0, 1.0)


def map_speed(speed_label: str) -> float:
    return MOVEMENT_SPEED_TPS.get(speed_label, MOVEMENT_SPEED_TPS["Medium"])


@dataclass
class StatRanges:
    health: Tuple[float, float] = (200.0, 5_000.0)
    damage: Tuple[float, float] = (80.0, 1_320.0)
    hit_speed: Tuple[float, float] = (0.7, 3.3)
    range: Tuple[float, float] = (1.0, 10.0)
    move_speed: Tuple[float, float] = (1.1, 3.9)
    lifetime: Tuple[float, float] = (30.0, 70.0)
    spell_duration: Tuple[float, float] = (0.0, 8.0)
    tile_radius: Tuple[float, float] = (1.0, 8.0)


# ===============
# Card formulas
# ===============
def card_overall(card: Card, ranges: StatRanges) -> float:
    if card.card_type == CardType.TOWER_ATTACKER:
        h = linear_norm(card.health, *ranges.health)
        d = linear_norm(card.damage, *ranges.damage)
        r = linear_norm(card.attack_range, *ranges.range)
        ms = linear_norm(map_speed(card.move_speed), *ranges.move_speed)
        hs = inverse_norm(card.hit_speed, *ranges.hit_speed)
        target_bonus = 0.05 if card.targets.lower() == "towers only" else 0.0
        score = 0.25 * h + 0.30 * d + 0.20 * r + 0.10 * ms + 0.15 * hs + target_bonus
        return round(min(1.0, score) * 100, 2)

    if card.card_type == CardType.BUILDING:
        h = linear_norm(card.health, *ranges.health)
        d = linear_norm(card.damage, *ranges.damage)
        r = linear_norm(card.attack_range, *ranges.range)
        lt = linear_norm(card.lifetime, *ranges.lifetime)
        hs = inverse_norm(card.hit_speed, *ranges.hit_speed)
        score = 0.25 * h + 0.30 * d + 0.20 * r + 0.10 * lt + 0.15 * hs
        return round(score * 100, 2)

    if card.card_type == CardType.SPELL:
        d = linear_norm(card.damage, *ranges.damage)
        sd_inv = inverse_norm(card.spell_duration, *ranges.spell_duration)
        tr = linear_norm(card.tile_radius, *ranges.tile_radius)
        score = 0.50 * d + 0.30 * sd_inv + 0.20 * tr
        return round(score * 100, 2)

    # Troop baseline for all air/ground troop archetypes
    h = linear_norm(card.health, *ranges.health)
    d = linear_norm(card.damage, *ranges.damage)
    hs = inverse_norm(card.hit_speed, *ranges.hit_speed)
    ms = linear_norm(map_speed(card.move_speed), *ranges.move_speed)
    r = linear_norm(card.attack_range, *ranges.range)
    score = 0.25 * h + 0.23 * d + 0.16 * hs + 0.14 * ms + 0.22 * r
    return round(score * 100, 2)


def usage_weights(cards: Sequence[Card]) -> Dict[str, float]:
    raw = {c.name: 4 / c.elixir_cost for c in cards}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def update_death_score(card: Card, team_matches: int) -> float:
    usage_pct = card.matches_used / team_matches if team_matches else 0.0
    card.death_score = clamp(card.death_score + (0.3 - usage_pct), -5.0, 5.0)
    return card.death_score


def card_is_dead(card: Card) -> bool:
    return card.death_score >= 3.0


# =================
# Team-level formulas
# =================
def team_overall(deck: Deck, ranges: StatRanges) -> float:
    deck.validate()
    return round(sum(card_overall(tc.card, ranges) for tc in deck.active) / 4, 2)


def chemistry_score(deck: Deck) -> float:
    # Pairwise archetype synergy: same family +1, conflicting air-ground tower focus -1.
    synergy = 0
    for a, b in combinations([tc.card for tc in deck.active], 2):
        a_prefix = a.archetype.split()[0]
        b_prefix = b.archetype.split()[0]
        if a_prefix == b_prefix:
            synergy += 1
        if "Tower" in a.archetype and "Tower" not in b.archetype:
            synergy -= 1
    synergy = max(-4, min(4, synergy))
    history = max(0, min(3, int(sum(tc.card.seasons_together for tc in deck.active) / 4)))
    chemistry = 0.6 * synergy + 0.4 * history
    return chemistry


def chemistry_modifier(deck: Deck) -> float:
    return chemistry_score(deck) / 10


def counter_modifier(counter_weight: float, has_advantage: bool) -> float:
    sign = 1 if has_advantage else -1
    return sign * 0.06 * counter_weight


def streak_modifier(streak: int) -> float:
    s = max(-5, min(5, streak))
    return STREAK_TABLE[s]


def home_bonus(crystals: int) -> float:
    return 0.01 + 0.02 * (clamp(crystals, 0, MAX_CRYSTALS) / MAX_CRYSTALS)


def strategy_modifier(good: bool, bad: bool) -> float:
    if good:
        return 0.03
    if bad:
        return -0.03
    return 0.0


def identity_strength_modifier(identity: TeamIdentity) -> float:
    # Strength-impacting part only.
    if identity == TeamIdentity.DEFENSIVE:
        return 0.02  # converted from KO resistance abstraction
    if identity == TeamIdentity.CONTROL:
        return 0.01  # slight tactical strength
    return 0.0


def compute_strength_pre_rng(
    base_strength: float,
    counter_mod: float,
    chem_mod: float,
    identity_mod: float,
    streak_mod: float,
    home_mod: float,
    strategy_mod_value: float,
) -> float:
    return base_strength * (1 + counter_mod + chem_mod + identity_mod + streak_mod + home_mod + strategy_mod_value)


def final_strength(strength_pre_rng: float, phase_rng: float) -> float:
    return (0.78 * strength_pre_rng) + (0.22 * phase_rng)


@dataclass
class MatchResult:
    home_team: str
    away_team: str
    winner: str
    loser: str
    home_score: float
    away_score: float
    duration_seconds: int
    home_crowns: int
    away_crowns: int
    home_ko: bool
    away_ko: bool
    home_deep_score: int
    away_deep_score: int
    home_crystals_delta: int
    away_crystals_delta: int


class MatchEngine:
    phase_weights = [0.30, 0.30, 0.40]

    def __init__(self, season_seed: int, ranges: Optional[StatRanges] = None):
        self.seed = season_seed
        self.rng = random.Random(season_seed)
        self.ranges = ranges or StatRanges()

    def _micro_variance(self) -> float:
        mag = self.rng.uniform(0.01, 0.02)
        sign = -1 if self.rng.random() < 0.5 else 1
        return 1 + sign * mag

    def _phase_rng(self, home_identity: TeamIdentity, away_identity: TeamIdentity) -> Tuple[float, float]:
        # Tempo identity reduces rng swing by 20%
        base_home = self.rng.uniform(-0.11, 0.11)
        base_away = self.rng.uniform(-0.11, 0.11)
        if home_identity == TeamIdentity.TEMPO:
            base_home *= 0.8
        if away_identity == TeamIdentity.TEMPO:
            base_away *= 0.8
        return base_home, base_away

    def simulate_match(
        self,
        home: Team,
        away: Team,
        home_counter_weight: float = 0.0,
        away_counter_weight: float = 0.0,
        home_has_counter_advantage: bool = False,
        away_has_counter_advantage: bool = False,
        home_strategy_good: bool = False,
        away_strategy_good: bool = False,
    ) -> MatchResult:
        home_base = team_overall(home.deck, self.ranges)
        away_base = team_overall(away.deck, self.ranges)

        home_pre = compute_strength_pre_rng(
            base_strength=home_base,
            counter_mod=counter_modifier(home_counter_weight, home_has_counter_advantage),
            chem_mod=chemistry_modifier(home.deck),
            identity_mod=identity_strength_modifier(home.identity),
            streak_mod=streak_modifier(home.streak),
            home_mod=home_bonus(home.crystals),
            strategy_mod_value=strategy_modifier(home_strategy_good, False),
        )
        away_pre = compute_strength_pre_rng(
            base_strength=away_base,
            counter_mod=counter_modifier(away_counter_weight, away_has_counter_advantage),
            chem_mod=chemistry_modifier(away.deck),
            identity_mod=identity_strength_modifier(away.identity),
            streak_mod=streak_modifier(away.streak),
            home_mod=0.0,
            strategy_mod_value=strategy_modifier(away_strategy_good, False),
        )

        micro = self._micro_variance()
        home_total = 0.0
        away_total = 0.0
        for w in self.phase_weights:
            home_rng, away_rng = self._phase_rng(home.identity, away.identity)
            home_total += final_strength(home_pre * micro, home_rng) * w
            away_total += final_strength(away_pre * micro, away_rng) * w

        winner, loser = (home, away) if home_total >= away_total else (away, home)
        margin = abs(home_total - away_total) / max(home_total, away_total, 1)

        duration = int(clamp(180 - (margin * 12), 30, 180))

        winner_crowns = 3 if margin >= 0.25 else 2 if margin >= 0.15 else 1 if margin >= 0.05 else 0
        loser_crowns = max(0, winner_crowns - 1)

        # Identity crown transforms
        if winner.identity == TeamIdentity.AGGRESSIVE:
            winner_crowns = min(3, int(round(winner_crowns * 1.04)))
        if winner.identity == TeamIdentity.DEFENSIVE:
            winner_crowns = max(0, int(round(winner_crowns * 0.97)))

        home_crowns = winner_crowns if winner is home else loser_crowns
        away_crowns = winner_crowns if winner is away else loser_crowns

        # KO logic
        ko_chance = 0.04
        if winner.identity == TeamIdentity.CONTROL:
            ko_chance += 0.06
        if loser.identity == TeamIdentity.DEFENSIVE:
            ko_chance -= 0.05
        ko_roll = self.rng.random() < max(0, ko_chance)

        home_ko = ko_roll and winner is home
        away_ko = ko_roll and winner is away

        home_deep = deep_score(win=winner is home, duration=duration, crowns=home_crowns, ko=home_ko)
        away_deep = deep_score(win=winner is away, duration=duration, crowns=away_crowns, ko=away_ko)

        home_delta, away_delta = crystal_deltas(home, away, winner is home, self.rng)

        apply_team_outcome(home, winner is home, home_delta, home_deep)
        apply_team_outcome(away, winner is away, away_delta, away_deep)

        return MatchResult(
            home_team=home.name,
            away_team=away.name,
            winner=winner.name,
            loser=loser.name,
            home_score=round(home_total, 3),
            away_score=round(away_total, 3),
            duration_seconds=duration,
            home_crowns=home_crowns,
            away_crowns=away_crowns,
            home_ko=home_ko,
            away_ko=away_ko,
            home_deep_score=home_deep,
            away_deep_score=away_deep,
            home_crystals_delta=home_delta,
            away_crystals_delta=away_delta,
        )


def time_bonus(win: bool, duration: int) -> int:
    table = WIN_TIME_BONUS if win else LOSS_TIME_BONUS
    for sec, bonus in table:
        if duration <= sec:
            return bonus
    return table[-1][1]


def deep_score(win: bool, duration: int, crowns: int, ko: bool) -> int:
    ko_bonus = 10 if (win and ko) else -10 if (not win and ko) else 0
    return time_bonus(win, duration) + ko_bonus + CROWN_BONUS.get(crowns, 0)


def crystal_deltas(home: Team, away: Team, home_wins: bool, rng: random.Random) -> Tuple[int, int]:
    base_win = rng.randint(490, 525)
    base_loss = -rng.randint(480, 515)

    # Delta PR = (opp - self)/100
    home_delta_pr = (away.power_rank - home.power_rank) / 100
    away_delta_pr = (home.power_rank - away.power_rank) / 100

    if home_wins:
        home_scaled = int(round(base_win * (1 + 0.15 * home_delta_pr)))
        away_scaled = int(round(base_loss * (1 + 0.15 * away_delta_pr)))
    else:
        away_scaled = int(round(base_win * (1 + 0.15 * away_delta_pr)))
        home_scaled = int(round(base_loss * (1 + 0.15 * home_delta_pr)))

    return home_scaled, away_scaled


def apply_team_outcome(team: Team, won: bool, crystals_delta: int, deep: int) -> None:
    if won:
        team.wins += 1
        team.streak = team.streak + 1 if team.streak >= 0 else 1
    else:
        team.losses += 1
        team.streak = team.streak - 1 if team.streak <= 0 else -1
    team.crystals = int(clamp(team.crystals + crystals_delta, 0, MAX_CRYSTALS))
    team.total_deep_score += deep


# =================
# Contracts and PR
# =================
def salary_from_overall(overall: float) -> int:
    salary = MIN_SALARY + (overall / 100) * (MAX_SALARY - MIN_SALARY)
    return int(round(salary))


def power_rank(teams: Sequence[Team]) -> Dict[str, float]:
    crystals = [t.crystals for t in teams]
    win_pcts = [t.win_pct for t in teams]
    deep_scores = [t.avg_deep_score for t in teams]

    c_min, c_max = min(crystals), max(crystals)
    w_min, w_max = min(win_pcts), max(win_pcts)
    s_min, s_max = min(deep_scores), max(deep_scores)

    out: Dict[str, float] = {}
    for t in teams:
        c_n = linear_norm(t.crystals, c_min, c_max)
        w_n = linear_norm(t.win_pct, w_min, w_max)
        s_n = linear_norm(t.avg_deep_score, s_min, s_max)
        pr = (0.5 * c_n + 0.3 * w_n + 0.2 * s_n) * 100
        t.power_rank = round(pr, 2)
        out[t.name] = t.power_rank
    return out


# ===============
# AI systems
# ===============
def trade_value(card: Card, overall: float, salary: int, scarcity: float) -> float:
    salary_eff = (MAX_SALARY - salary) / (MAX_SALARY - MIN_SALARY)
    death_penalty = max(0.0, card.death_score / 5)
    return (0.55 * overall) + (0.25 * scarcity * 100) + (0.20 * salary_eff * 100) - (death_penalty * 10)


def ai_assign_identity(deck: Deck) -> TeamIdentity:
    cards = [tc.card for tc in deck.active]
    tower_attackers = sum(c.card_type == CardType.TOWER_ATTACKER for c in cards)
    buildings = sum(c.card_type == CardType.BUILDING for c in cards)
    spells = sum(c.card_type == CardType.SPELL for c in cards)

    if tower_attackers >= 2:
        return TeamIdentity.AGGRESSIVE
    if buildings >= 2:
        return TeamIdentity.DEFENSIVE
    if spells >= 2:
        return TeamIdentity.CONTROL
    return TeamIdentity.TEMPO


def ai_build_deck(card_pool: Sequence[Card], ranges: StatRanges, budget: int = SALARY_CAP) -> Deck:
    cards = [c for c in card_pool if not card_is_dead(c)]
    scores = sorted(cards, key=lambda c: card_overall(c, ranges), reverse=True)

    active: List[TeamCard] = []
    reserves: List[TeamCard] = []

    for c in scores:
        overall = card_overall(c, ranges)
        salary = salary_from_overall(overall)
        if sum(tc.contract.salary for tc in active + reserves) + salary > budget:
            continue
        tc = TeamCard(c, Contract(salary=salary, years=3))

        if len(active) < 4:
            active.append(tc)
        elif len(reserves) < 2:
            reserves.append(tc)

        if len(active) == 4 and len(reserves) == 2:
            break

    deck = Deck(active=active, reserves=reserves)
    deck.validate()
    return deck


def evaluate_trade(incoming_value: float, outgoing_value: float, rng: random.Random, deadline_week: bool = False) -> bool:
    net = incoming_value - outgoing_value
    threshold = 5
    if deadline_week:
        threshold *= 0.75

    if net >= threshold:
        return True
    if net <= -threshold:
        return False
    return rng.random() < 0.20


def offer_score(salary_n: float, team_pr_n: float, role_score: float) -> float:
    return 0.65 * salary_n + 0.25 * team_pr_n + 0.10 * role_score


def expansion_trigger(card_pool_size: int, avg_cards_per_team: float) -> bool:
    return card_pool_size > 70 and avg_cards_per_team < 3.8


# =====================
# Schedules and playoffs
# =====================
@dataclass
class ScheduledMatch:
    week: int
    home: str
    away: str


class SeasonScheduler:
    def __init__(self, season_seed: int):
        self.rng = random.Random(season_seed)

    def generate_regular_season(self, dml: Sequence[Team], msl: Sequence[Team]) -> List[ScheduledMatch]:
        d_names = [t.name for t in dml]
        m_names = [t.name for t in msl]
        schedule: List[ScheduledMatch] = []

        week = 1
        # Conference games
        for a, b in combinations(d_names, 2):
            for _ in range(4):
                home, away = (a, b) if self.rng.random() < 0.5 else (b, a)
                schedule.append(ScheduledMatch(week=week, home=home, away=away))
                week = 1 + (week % 12)

        for a, b in combinations(m_names, 2):
            reps = 4 if self.rng.random() < 0.5 else 3
            for _ in range(reps):
                home, away = (a, b) if self.rng.random() < 0.5 else (b, a)
                schedule.append(ScheduledMatch(week=week, home=home, away=away))
                week = 1 + (week % 12)

        # Cross-conference with max 3 repeats per pair
        pair_counts: Dict[Tuple[str, str], int] = {}
        for d in d_names:
            for _ in range(13):
                options = [m for m in m_names if pair_counts.get((d, m), 0) < 3]
                opp = self.rng.choice(options)
                pair_counts[(d, opp)] = pair_counts.get((d, opp), 0) + 1
                home, away = (d, opp) if self.rng.random() < 0.5 else (opp, d)
                schedule.append(ScheduledMatch(week=week, home=home, away=away))
                week = 1 + (week % 12)

        return schedule


def tiebreak_sort(teams: Sequence[Team], rng: random.Random) -> List[Team]:
    def key(t: Team):
        return (t.crystals, t.total_deep_score, t.wins, t.power_rank, -t.losses)

    grouped = sorted(teams, key=key, reverse=True)
    i = 0
    while i < len(grouped) - 1:
        if key(grouped[i]) == key(grouped[i + 1]):
            if rng.random() < 0.5:
                grouped[i], grouped[i + 1] = grouped[i + 1], grouped[i]
        i += 1
    return grouped


# ======================
# Bootstrapped card pool
# ======================
def starter_card_pool() -> List[Card]:
    # Includes all concrete cards supplied in the pasted document excerpt.
    data = [
        # Ground Ranged
        ("Tactical Sniper", "Ground Ranged", CardType.TROOP, 890, 612, 2.1, "Slow", 10, 0, 0, 0, 5, "Ground Troops & Buildings"),
        ("Archer", "Ground Ranged", CardType.TROOP, 512, 144, 1.1, "Medium", 6, 0, 0, 0, 3, "Ground Troops & Buildings"),
        ("Recon Hunter", "Ground Ranged", CardType.TROOP, 1044, 233, 1.4, "Fast", 7, 0, 0, 0, 4, "Ground Troops & Buildings"),
        ("Combat Sharpshooter", "Ground Ranged", CardType.TROOP, 1210, 478, 1.8, "Medium", 8, 0, 0, 0, 5, "Ground Troops & Buildings"),
        ("Precision Ranger", "Ground Ranged", CardType.TROOP, 932, 355, 1.2, "Fast", 9, 0, 0, 0, 4, "Ground Troops & Buildings"),
        # Ground Tower Attacker
        ("Fort-Rammer", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 4210, 622, 2.8, "Slow", 1, 0, 0, 0, 6, "Towers Only"),
        ("Demolition Bull", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 3550, 744, 2.4, "Medium", 1, 0, 0, 0, 6, "Towers Only"),
        ("Rampage Juggernaut", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 4980, 910, 3.1, "Very Slow", 1, 0, 0, 0, 8, "Towers Only"),
        ("Iron Marauder", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 2870, 520, 2.2, "Medium", 1, 0, 0, 0, 5, "Towers Only"),
        ("Citadel Crusher", "Ground Tower Attacker", CardType.TOWER_ATTACKER, 4620, 820, 3.3, "Slow", 1, 0, 0, 0, 7, "Towers Only"),
        # Buildings
        ("Missile Silo", "Building", CardType.BUILDING, 3400, 510, 1.7, "Very Slow", 8, 60, 0, 0, 5, "Ground Troops & Buildings"),
        ("Tesla Tower", "Building", CardType.BUILDING, 2900, 380, 1.3, "Very Slow", 6, 55, 0, 0, 4, "Ground Troops & Buildings"),
        ("Drone Control Post", "Building", CardType.BUILDING, 2400, 290, 1.2, "Very Slow", 7, 65, 0, 0, 4, "Ground Troops & Buildings"),
        ("Turret", "Building", CardType.BUILDING, 1800, 255, 1.1, "Very Slow", 5, 50, 0, 0, 3, "Ground Troops & Buildings"),
        ("Bunker Buster", "Building", CardType.BUILDING, 4200, 610, 1.9, "Very Slow", 4, 45, 0, 0, 6, "Ground Troops & Buildings"),
        # Spells
        ("Sandstorm", "Spell", CardType.SPELL, 0, 480, 1.0, "Medium", 0, 0, 6, 5, 4, "All"),
        ("Acid Rain", "Spell", CardType.SPELL, 0, 610, 1.0, "Medium", 0, 0, 7, 4, 5, "All"),
        ("Gravity Pulse", "Spell", CardType.SPELL, 0, 300, 1.0, "Medium", 0, 0, 5, 6, 3, "All"),
        ("Nova Shock", "Spell", CardType.SPELL, 0, 880, 1.0, "Medium", 0, 0, 0, 3, 6, "All"),
        ("Meteor Shower", "Spell", CardType.SPELL, 0, 1320, 1.0, "Medium", 0, 0, 8, 8, 8, "All"),
        # Added explicit Battle Trooper request line
        ("Battle Trooper", "Ground Melee Splash", CardType.TROOP, 2199, 110, 0.8, "Medium", 1.5, 0, 0, 0, 4, "Ground Troops & Buildings"),
    ]

    return [
        Card(
            name=n,
            archetype=a,
            card_type=t,
            health=hp,
            damage=dmg,
            hit_speed=hs,
            move_speed=spd,
            attack_range=rng,
            lifetime=lt,
            spell_duration=sd,
            tile_radius=tr,
            elixir_cost=el,
            targets=targets,
        )
        for n, a, t, hp, dmg, hs, spd, rng, lt, sd, tr, el, targets in data
    ]


def build_example_league(seed: int = 42) -> List[Team]:
    rng = random.Random(seed)
    ranges = StatRanges()
    cards = starter_card_pool()
    rng.shuffle(cards)

    team_names = [
        "Orion", "Nebula", "Vortex", "Quasar", "Nova", "Pulse", "Aether", "Helix",
        "Ember", "Frost", "Tide", "Volt", "Echo", "Drift", "Zenith"
    ]

    teams: List[Team] = []
    for i, name in enumerate(team_names):
        pool = cards[:] if len(cards) >= 6 else cards * 2
        rng.shuffle(pool)
        deck = ai_build_deck(pool, ranges)
        team = Team(name=name, conference="DML" if i < 8 else "MSL", deck=deck)
        team.identity = ai_assign_identity(team.deck)
        teams.append(team)

    power_rank(teams)
    return teams


def run_demo_season(seed: int = 42) -> Dict[str, object]:
    teams = build_example_league(seed)
    team_map = {t.name: t for t in teams}
    dml = [t for t in teams if t.conference == "DML"]
    msl = [t for t in teams if t.conference == "MSL"]

    scheduler = SeasonScheduler(seed)
    schedule = scheduler.generate_regular_season(dml, msl)

    engine = MatchEngine(seed)
    for sm in schedule[:200]:  # bounded for fast simulation/demo
        home = team_map[sm.home]
        away = team_map[sm.away]
        engine.simulate_match(home, away)

    power_rank(teams)

    standings = {
        "DML": [t.name for t in tiebreak_sort(dml, random.Random(seed))],
        "MSL": [t.name for t in tiebreak_sort(msl, random.Random(seed + 1))],
    }

    return {
        "teams": teams,
        "standings": standings,
        "schedule_size": len(schedule),
    }


if __name__ == "__main__":
    summary = run_demo_season(2025)
    print(f"Simulated schedule entries: {summary['schedule_size']}")
    print("DML top 4:", summary["standings"]["DML"][:4])
    print("MSL top 4:", summary["standings"]["MSL"][:4])
