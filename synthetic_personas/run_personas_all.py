"""
python -m synthetic_personas.mass_replay
"""
import glob
import os

from synthetic_personas.replay_pipeline import run_replay

dir = "synthetic_transcripts"

def main():
    for path in sorted(glob.glob(os.path.join(dir, "*.json"))):
        print(f"\n=== Replaying {path} ===")
        try:
            run_replay(path)
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    main()
