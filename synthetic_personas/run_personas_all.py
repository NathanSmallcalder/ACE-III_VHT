"""
python -m synthetic_personas.mass_replay
"""
import glob
import os

from synthetic_personas.replay_pipeline import run_replay

dir = "synthetic_transcripts"

def main():
    for path in sorted(glob.glob(os.path.join(dir, "*.json"))):
        participant_id = os.path.splitext(os.path.basename(path))[0]
        if glob.glob(os.path.join("results", f"ACE-III_{participant_id}_*.json")):
            print(f"\n=== Skipping {path} ===")
            continue
        try:
            run_replay(path)
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    main()
