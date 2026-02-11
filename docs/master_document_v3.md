# Battle Arena League — Master Document V3 (Draft Integration)

This document consolidates the rules, formulas, and starter card data provided for the Battle Arena League simulation model.

## 1) Card System Overview

- **Total cards:** 60
- **Archetypes:** 12
- **Card types:** Troops, Tower Attackers, Buildings, Spells
- **Deck structure:**
  - 4 active cards
  - 0–2 reserves
  - max deck size: 6
- Cards never age out automatically. Cards leave only if manually removed.

## 2) Archetypes

- Ground Melee
- Ground Melee Splash
- Ground Ranged
- Ground Ranged Splash
- Air Melee
- Air Melee Splash
- Air Ranged
- Air Ranged Splash
- Specialized
- Ground Tower Attacker
- Air Tower Attacker
- Building
- Spell

## 3) Movement Speed Mapping (tiles/sec)

- Very Slow = 1.1
- Slow = 1.8
- Medium = 2.5
- Fast = 3.2
- Very Fast = 3.9

## 4) Normalization Functions

### 4.1 Linear Normalization (higher is better)

\[
\mathrm{norm}(x)=\frac{x-\mathrm{min}}{\mathrm{max}-\mathrm{min}}
\]

### 4.2 Inverse Normalization (lower is better)

\[
\mathrm{norm_{inv}}(x)=\frac{\mathrm{max}-x}{\mathrm{max}-\mathrm{min}}
\]

### 4.3 Clamp

\[
0\leq \mathrm{norm}\leq 1
\]

## 5) Card Overall Formulas (0–100)

All final overalls are:

\[
\mathrm{Overall}=\mathrm{Weighted\ Score}\times 100
\]

### 5.1 Tower Attacker

- Health: 0.25
- Damage: 0.30
- Range: 0.20
- Move speed: 0.10
- Hit speed (inverse): 0.15
- Targets = Towers only: +0.05 bonus (cap 1.0)

\[
\mathrm{Score}=0.25H+0.30D+0.20R+0.10MS+0.15HS+\mathrm{TargetBonus}
\]
\[
\mathrm{Overall}=\min(1.0,\mathrm{Score})\times 100
\]

### 5.2 Building

- Health: 0.25
- Damage: 0.30
- Range: 0.20
- Lifetime: 0.10
- Hit speed (inverse): 0.15

\[
\mathrm{Score}=0.25H+0.30D+0.20R+0.10LT+0.15HS
\]
\[
\mathrm{Overall}=\mathrm{Score}\times 100
\]

### 5.3 Spell

- Damage: 0.50
- Spell duration (inverse): 0.30
- Tile radius: 0.20

\[
\mathrm{Score}=0.50D+0.30SD_{inv}+0.20TR
\]
\[
\mathrm{Overall}=\mathrm{Score}\times 100
\]

### 5.4 Troop (air & ground)

- Health: 0.25
- Damage: 0.23
- Hit speed (inverse): 0.16
- Movement speed: 0.14
- Range: 0.22
- ATK type: tag only (0)

\[
\mathrm{Score}=0.25H+0.23D+0.16HS+0.14MS+0.22R
\]
\[
\mathrm{Overall}=\mathrm{Score}\times 100
\]

## 6) Team Overall Formula

Using 4 active cards:

\[
\mathrm{Team\ Overall}=\frac{\sum_{i=1}^{4}\mathrm{CardOverall}_i}{4}
\]

## 7) Deck Validation Rules

- Exactly 4 active cards
- 0–2 reserves
- Max deck size = 6

## 8) Usage Weighting

\[
\mathrm{UsageWeight}=\frac{4}{\mathrm{ElixirCost}}
\]

Normalize all card usage weights to sum to 1.0.

## 9) Card Death Logic (Meta Decay)

Cards never retire automatically.

### 9.1 Usage %

\[
\mathrm{Usage\%}=\frac{\mathrm{MatchesUsed}}{\mathrm{TeamMatches}}
\]

### 9.2 Death score

\[
\mathrm{DeathScore}_{new}=\mathrm{DeathScore}_{old}+(0.3-\mathrm{Usage\%})
\]

Clamp to \([-5, 5]\).

### 9.3 Death threshold

If DeathScore ≥ 3:
- AI stops using/signing card
- card is considered dead in meta
- user can still revive/remove manually

## 10) Match Engine (3-Phase)

- Opening: 0–60s (30%)
- Midgame: 60–120s (30%)
- Double Elixir: 120–180s (40%)

Deterministic season-level simulation with per-match micro-variance.

### 10.1 RNG

- Season seed initializes deterministic RNG.
- Match micro-variance: \(1\pm 0.01\) to \(1\pm 0.02\)
- Phase RNG: uniform \([-0.11, +0.11]\)

### 10.2 Team strength

\[
\mathrm{Strength}_{preRNG}=\mathrm{BaseStrength}\times(1+\mathrm{CounterMod}+\mathrm{ChemMod}+\mathrm{IdentityMod}+\mathrm{StreakMod}+\mathrm{HomeMod}+\mathrm{StrategyMod})
\]
\[
\mathrm{FinalStrength}=(0.78\times\mathrm{Strength}_{preRNG})+(0.22\times\mathrm{PhaseRNG})
\]

### 10.3 Phase score

\[
\mathrm{PhaseScore}=\mathrm{FinalStrength}\times\mathrm{PhaseWeight}
\]

### 10.4 Match outputs

- Winner from total phase score
- Duration: \(180-(\mathrm{Margin}\times 12)\), clamp [30,180]
- Crowns by margin bands: 0/1/2/3
- KO logic and deep-score bonuses/penalties applied

## 11) Season Structure

- Regular season: 41 games/team
- All-Star break
- Playoffs: 8-team hybrid double-elim format
- Awards, off-season, free agency, patching, optional expansion

## 12) Contracts / Cap / Rank / Crystals

- Salary range: 23,400 to 198,200
- Salary formula:

\[
\mathrm{Salary}=23400+\left(\frac{\mathrm{Overall}}{100}\right)(198200-23400)
\]

- Cap: 500,000 hard cap
- Power rank components:
  - Crystals 0.50
  - Win% 0.30
  - Avg Deep Score 0.20

## 13) Starter Cards (Provided Data)

### 13.1 Ground Ranged

| Card | HP | DMG | HS | SPD | RNG | Elixir | Overall |
|---|---:|---:|---:|---|---:|---:|---:|
| Tactical Sniper | 890 | 612 | 2.1 | Slow | 10 | 5 | 78.4 |
| Archer | 512 | 144 | 1.1 | Medium | 6 | 3 | 62.7 |
| Recon Hunter | 1044 | 233 | 1.4 | Fast | 7 | 4 | 69.9 |
| Combat Sharpshooter | 1210 | 478 | 1.8 | Medium | 8 | 5 | 74.3 |
| Precision Ranger | 932 | 355 | 1.2 | Fast | 9 | 4 | 76.1 |

### 13.2 Ground Ranged Splash

| Card | HP | DMG | HS | SPD | RNG | Elixir | Overall |
|---|---:|---:|---:|---|---:|---:|---:|
| Combat Bomber | 980 | 402 | 2.4 | Slow | 6 | 5 | 71.0 |
| Blast Engineer | 1212 | 318 | 1.9 | Medium | 5 | 4 | 68.2 |
| Siege Specialist | 1830 | 544 | 2.6 | Very Slow | 7 | 6 | 76.9 |
| Shockwave Guards | 1420 | 265 | 1.5 | Medium | 4 | 4 | 64.4 |
| Fireburst Soldier | 860 | 390 | 1.3 | Fast | 6 | 4 | 70.7 |

### 13.3 Ground Tower Attacker

| Card | HP | DMG | HS | SPD | RNG | Elixir | Overall |
|---|---:|---:|---:|---|---:|---:|---:|
| Fort-Rammer | 4210 | 622 | 2.8 | Slow | 1 | 6 | 81.3 |
| Demolition Bull | 3550 | 744 | 2.4 | Medium | 1 | 6 | 83.6 |
| Rampage Juggernaut | 4980 | 910 | 3.1 | Very Slow | 1 | 8 | 88.9 |
| Iron Marauder | 2870 | 520 | 2.2 | Medium | 1 | 5 | 77.2 |
| Citadel Crusher | 4620 | 820 | 3.3 | Slow | 1 | 7 | 86.1 |

### 13.4 Buildings

| Card | HP | DMG | RNG | LT | Overall |
|---|---:|---:|---:|---:|---:|
| Missile Silo | 3400 | 510 | 8 | 60 | 79.4 |
| Tesla Tower | 2900 | 380 | 6 | 55 | 72.6 |
| Drone Control Post | 2400 | 290 | 7 | 65 | 70.1 |
| Turret | 1800 | 255 | 5 | 50 | 63.8 |
| Bunker Buster | 4200 | 610 | 4 | 45 | 80.9 |

### 13.5 Air Melee

| Card | HP | DMG | HS | SPD | RNG | Elixir | Overall |
|---|---:|---:|---:|---|---:|---:|---:|
| Sky Lion | 2120 | 334 | 1.3 | Fast | 1 | 4 | 70.4 |
| Storm Hawk | 1640 | 290 | 1.1 | Very Fast | 1 | 3 | 69.7 |
| Aero Jet | 980 | 412 | 0.9 | Very Fast | 1 | 4 | 73.2 |
| Phantom | 740 | 520 | 0.7 | Very Fast | 1 | 4 | 75.8 |
| War Griffin | 2780 | 388 | 1.6 | Medium | 1 | 5 | 74.1 |

### 13.6 Spells

| Card | DMG | Radius | Duration | Overall |
|---|---:|---:|---|---:|
| Sandstorm | 480 | 5 | 6s | 71.5 |
| Acid Rain | 610 | 4 | 7s | 76.8 |
| Gravity Pulse | 300 | 6 | 5s | 69.2 |
| Nova Shock | 880 | 3 | instant | 82.1 |
| Meteor Shower | 1320 | 8 | 8s | 90.3 |

### 13.7 Air Ranged

| Card | HP | DMG | HS | SPD | RNG | Overall |
|---|---:|---:|---:|---|---:|---:|
| Jet Marksman | 1020 | 412 | 1.5 | Fast | 8 | 75.4 |
| Recon Scout | 740 | 266 | 1.1 | Very Fast | 7 | 69.1 |
| Precision Hawk | 960 | 344 | 1.2 | Fast | 9 | 74.8 |
| Gunner | 1320 | 288 | 0.9 | Medium | 6 | 71.0 |
| Magic Archer | 860 | 380 | 1.4 | Fast | 10 | 78.2 |

### 13.8 Air Ranged Splash

| Card | HP | DMG | HS | SPD | RNG | Overall |
|---|---:|---:|---:|---|---:|---:|
| Rocka-Copter | 1840 | 488 | 2.3 | Slow | 6 | 76.3 |
| Assault Zeppelin | 3200 | 612 | 2.8 | Very Slow | 7 | 82.6 |
| Tactical Bomber | 1460 | 520 | 2.0 | Medium | 6 | 77.8 |
| Plasma Chopper | 1720 | 444 | 1.6 | Fast | 5 | 74.9 |
| Atomic Precision | 920 | 880 | 2.5 | Medium | 9 | 88.1 |

### 13.9 Air Tower Attacker

| Card | HP | DMG | HS | SPD | RNG | Overall |
|---|---:|---:|---:|---|---:|---:|
| Skybreaker | 2880 | 720 | 2.6 | Medium | 1 | 84.7 |
| Cloud Reaver | 2240 | 610 | 2.1 | Fast | 1 | 80.2 |
| Void Striker | 1820 | 840 | 2.9 | Fast | 1 | 86.4 |
| Aerial Juggernaut | 4120 | 910 | 3.2 | Slow | 1 | 90.1 |
| Storm Obliterator | 3550 | 780 | 2.8 | Medium | 1 | 87.9 |

### 13.10 Ground Melee (extended rows)

| Card Type | Card Name | Health | Damage | Hit Speed | Movement Speed | Attack Type | Tile Range | Targets | Elixir | Spawn Count | Splash Radius |
|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---|
| Ground Melee | Iron Soldier | 2321 | 231 | 1.2 | Medium | Melee | 1.5 | Ground Troops & Buildings | 3 | 1 | n/a |
| Ground Melee | Battle Knight | 1764 | 252 | 1.4 | Fast | Melee | 1 | Ground Troops & Buildings | 3 | 1 | n/a |
| Ground Melee | Riot Defender | 3218 | 342 | 1.2 | Slow | Melee | 2 | Ground Troops & Buildings | 6 | 1 | n/a |
| Ground Melee | Axe Warriors | 1432 | 231 | 1.1 | Medium | Melee | 1.5 | Ground Troops & Buildings | 7 | 2 | n/a |
| Ground Melee | Vanguardian | 890 | 132 | 0.7 | Very Fast | Melee | 1 | Ground Troops & Buildings | 2 | 1 | n/a |

### 13.11 Ground Melee Splash (extended rows)

| Card Type | Card Name | Health | Damage | Hit Speed | Movement Speed | Attack Type | Tile Range | Targets | Elixir | Spawn Count | Splash Radius |
|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---:|
| Ground Melee Splash | Thunder Crusher | 4296 | 523 | 2.1 | Very Slow | Melee Splash | 2 | Ground Troops & Buildings | 8 | 1 | 2.5 |
| Ground Melee Splash | Impact Guards | 233 | 89 | 1.0 | Fast | Melee Splash | 2.5 | Ground Troops & Buildings | 3 | 3 | 1.5 |
| Ground Melee Splash | S.H Destroyer | 2011 | 221 | 1.1 | Fast | Melee Splash | 2 | Ground Troops & Buildings | 5 | 1 | 3.0 |
| Ground Melee Splash | Titan Brawler | 1432 | 178 | 0.9 | Fast | Melee Splash | 1.5 | Ground Troops & Buildings | 5 | 1 | 2.0 |
| Ground Melee Splash | Battle Trooper | 2199 | 110 | 0.8 | Medium | Melee Splash | 1.5 | Ground Troops & Buildings | 4 | 1 | 2.5 |

---

Status: consolidated draft based on provided Section 2/3/4/5/7 data.
