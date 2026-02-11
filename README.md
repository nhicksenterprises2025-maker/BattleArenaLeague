# Battle Arena League Simulation App

This repository now contains a full Python simulation app implementing the Battle Arena League master-document systems:

- Card formulas (troop, tower attacker, building, spell)
- Deck validation (4 active, 0-2 reserve, max 6)
- Team overall formula
- Elixir usage weighting
- Card death-score meta decay
- Multi-phase match engine with seeded RNG
- Counter, chemistry, identity, streak, home, strategy modifiers
- Crowns, KO logic, deep score, crystals
- Contracts, salary cap math, power rank
- AI deck building, trade evaluation, free-agency offer scoring, expansion trigger
- Deterministic schedule generation for DML/MSL structures

## Run demo season

```bash
python battle_arena_league.py
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
