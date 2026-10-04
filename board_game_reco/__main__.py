from __future__ import annotations

import argparse
import json

from .recommender import Recommender


def main() -> None:
    parser = argparse.ArgumentParser(prog="board_game_reco", description="Recommend one board game.")
    parser.add_argument("text", help='for example: "family game for 4, not too complicated"')
    parser.add_argument("--trace", action="store_true", help="print the per-stage counts")
    args = parser.parse_args()

    recommender = Recommender.default()
    result = recommender.recommend(args.text)
    if result.game is not None:
        print(result.game.name)
        print(result.reason)
    if result.follow_up:
        print(result.follow_up)
    if result.game is None and not result.follow_up:
        print(result.reason)
    print("Understood by:", "rules, no AI call" if result.engine == "rules" else "Cloud AI")
    if args.trace:
        print(json.dumps(result.trace | {"terms": result.terms}, indent=2, default=str))
        if recommender.llm is not None:
            print("Tokens:", recommender.llm.usage)


if __name__ == "__main__":
    main()
