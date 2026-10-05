#!/bin/sh
# app: the Streamlit app. test: the offline test suite. perf: N3 latency with no fp32 download.
# cli: python -m board_game_reco. Anything else runs as a command.
set -eu
mode="${1:-app}"
[ "$#" -gt 0 ] && shift
case "$mode" in
  app)  exec streamlit run app/main.py --server.address=0.0.0.0 --server.port=8501 "$@" ;;
  test) exec python -m pytest -m "not network and not fp32 and not map" -p no:cacheprovider "$@" ;;
  perf) exec python -m evals.run perf --no-fp32 "$@" ;;
  cli)  exec python -m board_game_reco "$@" ;;
  *)    exec "$mode" "$@" ;;
esac
