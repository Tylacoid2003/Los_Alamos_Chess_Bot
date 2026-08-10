import subprocess
import os

class FairyClient:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine path '{engine_path}' does not exist.")

        self.engine_path = engine_path
        self.process = None

    def _send_command(self, command: str):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def _wait_for(self, target_string: str):
        while True:
            line = self.process.stdout.readline()
            if line.strip() == target_string:
                break

    def connect(self):
        """Spawns the engine subprocess and initializes the Los Alamos variant."""
        self.process = subprocess.Popen(
            self.engine_path, # find this specific executable file and run it in the background
            # This will allow us to communicate with the engine via stdin and stdout
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Treats inputs/outputs as strings instead of raw bytes
            bufsize=1   # Line-buffered so we don't get stuck waiting for chunks
        )

        self._send_command("uci")
        self._wait_for("uciok")

        self._send_command("setoption name UCI_Variant value losalamos")

        self._send_command("isready")
        self._wait_for("readyok")

    def analyze(self, fen: str, depth: int = 10) -> dict:
        # Tell the engine the board state 
        self._send_command(f"position fen {fen}")
        # Ask the engine to analyze the position to a certain depth
        self._send_command(f"go depth {depth}")

        best_move = None
        score_type = None
        score_value = None

        while True:
            line = self.process.stdout.readline().strip()

            if "info" in line and "score" in line:
                line_split = line.split()
                try:
                    score_type = line_split[line_split.index("score") + 1]
                    score_value = int(line_split[line_split.index("score") + 2])
                except (ValueError, IndexError):
                    pass # Handle cases where the expected format isn't met
            elif "bestmove" in line:
                line_split = line.split()
                if len(line_split) > 1:
                    best_move = line_split[1]
                break

        return {
            "best_move": best_move,
            "score_type": score_type,
            "score_value": score_value
        }

    def best_move(self, fen: str, depth: int = 10) -> str:
        answer = self.analyze(fen, depth)
        try:
            if answer["best_move"] is not None:
                return answer["best_move"]
        except KeyError:
            pass
        

    def evaluate(self, fen: str, depth: int = 10) -> int:
        answer = self.analyze(fen, depth)
        try:
            if answer["score_value"] is not None:
                return answer["score_value"]
        except KeyError:
            pass

    def quit(self):
        if self.process is not None:
            self._send_command("quit")
            self.process.communicate(timeout=5)  # Wait for the process to terminate
            self.process = None

    