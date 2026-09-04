import json
import os
import sys
from importlib import import_module

# fd 1 becomes stderr before the agent loads, so agent output can never reach the protocol stream
protocol = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

sys.path.insert(0, sys.argv[1])
agent = import_module("agent")

protocol.write(json.dumps({"ready": True}) + "\n")
protocol.flush()

for line in sys.stdin:
    request = json.loads(line)
    move = agent.get_move(request["fen"], request["time_left_ms"])
    protocol.write(json.dumps({"move": move}) + "\n")
    protocol.flush()
