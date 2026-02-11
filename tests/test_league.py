import unittest

from battle_arena_league import (
    Card,
    CardType,
    Deck,
    TeamCard,
    Contract,
    StatRanges,
    card_overall,
    salary_from_overall,
    usage_weights,
    home_bonus,
    run_demo_season,
)


class TestBattleArenaLeague(unittest.TestCase):
    def test_salary_bounds(self):
        self.assertEqual(salary_from_overall(0), 23400)
        self.assertEqual(salary_from_overall(100), 198200)

    def test_tower_attacker_bonus(self):
        ranges = StatRanges()
        base = Card(
            name="A",
            archetype="Ground Tower Attacker",
            card_type=CardType.TOWER_ATTACKER,
            health=3000,
            damage=700,
            hit_speed=2.5,
            move_speed="Medium",
            attack_range=1,
            elixir_cost=6,
            targets="Ground Troops & Buildings",
        )
        bonus = Card(**{**base.__dict__, "name": "B", "targets": "Towers Only"})
        self.assertGreater(card_overall(bonus, ranges), card_overall(base, ranges))

    def test_deck_rules(self):
        c = Card(name="C", archetype="Ground Ranged", card_type=CardType.TROOP)
        tc = TeamCard(card=c, contract=Contract(salary=23400, years=1))
        deck = Deck(active=[tc, tc, tc, tc], reserves=[tc, tc])
        deck.validate()  # should not raise

    def test_usage_weights_normalize(self):
        cards = [
            Card(name="A", archetype="x", card_type=CardType.TROOP, elixir_cost=2),
            Card(name="B", archetype="x", card_type=CardType.TROOP, elixir_cost=4),
        ]
        w = usage_weights(cards)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["A"], w["B"])

    def test_home_bonus_range(self):
        self.assertAlmostEqual(home_bonus(0), 0.01)
        self.assertAlmostEqual(home_bonus(20_000), 0.03)

    def test_demo_season_runs(self):
        out = run_demo_season(123)
        self.assertIn("standings", out)
        self.assertTrue(out["schedule_size"] > 0)


if __name__ == "__main__":
    unittest.main()
