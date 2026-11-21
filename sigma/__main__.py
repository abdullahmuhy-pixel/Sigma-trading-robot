"""python -m sigma            -> start the robot (web UI on http://localhost:8000)
python -m sigma backtest EURUSD H1 3000
python -m sigma analyze XAUUSD H4"""
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "backtest":
        from . import backtest, db
        db.connect()
        print(json.dumps(backtest.run(args[1], args[2] if len(args) > 2 else "H1",
                                      int(args[3]) if len(args) > 3 else 3000), indent=2, default=str))
        return
    if args and args[0] == "analyze":
        from . import agent, db, engine
        db.connect()
        print(json.dumps(agent.compact(engine.analyze(args[1], args[2] if len(args) > 2 else "H1")),
                         indent=2, default=str))
        return
    import uvicorn
    uvicorn.run("sigma.server:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
