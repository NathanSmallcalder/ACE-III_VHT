# ACE-III cognitive test automation

### Prerequisites

```bash
pip install -r requirements.txt
winget install Microsoft.OpenJDK.21
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet
```

The JDK is required by `language-tool-python`, which runs LanguageTool on a JVM to
grammar-check the writing task.

### Hardware and Local Requirements

**Local models.** Everything `main.py` needs is served locally by LM Studio on
`http://localhost:1234/v1`. Load both models before starting a session — the text model is
used for dialogue, and the vision model scores the drawing tasks:

| Role | Model | Set in |
| --- | --- | --- |
| Text (dialogue, answer extraction) | `google/gemma-4-e4b` | `LLM/LLM.py` |
| Vision (clock, cube, infinity, writing) | `qwen/qwen3-vl-8b` | `LLM/vlm.py` |

**Furhat robot.** Run the Furhat SDK Virtual Robot locally with the remote API enabled;
`main.py` connects to `127.0.0.1` at startup and speech is routed through it, so the session
will not start without it.

**Webcam.** Needed for the drawing tasks, which capture through `camera/capture.py` at device
index `0` — change `camera_index` if you have more than one camera. A mount pointed down at a
sheet of paper works best, but you can also just hold the finished drawing up to the camera.

### API Keys

`main.py` needs no API keys — it runs entirely against the local models above. Keys are only
used by the offline evaluation scripts, read from a `.env` file in the project root:

```
GOOGLE_API_KEY=...    # synthetic_personas/generate_transcript.py, test/model_comparison.py 
CLAUDE=...            # test/model_comparison.py, test/vlm_confusion.py (Anthropic key)
FURHAT_LOCAL_AUTH = ...     # Furhat connection locally

```

Add only the key for the scripts you intend to run.

### Session Configuration

`json/session_config.json` is a **required input, not sample data** — edit it before every
session. Answers for the Attention orientation items and the Memory retrograde items are
resolved from it at startup (`data_loader.resolve_dynamic_answers`), so stale values are
marked as wrong answers.

```json
{
    "location": {
        "number": "4",
        "street": "Hospital Road",
        "town": "canterbury",
        "county": "kent",
        "country": "England"
    },
    "patient": { "name": "nathan", "dob": "2909-09-09" },
    "assessor": "na",
    "current_uk_pm": "Andy Burnham",
    "current_us_president": "Donald Trump",
    "previous_uk_pm": "Keir Starmer",
    "previous_us_president": "Joe Biden"
}
```

| Field | Used for |
| --- | --- |
| `location.*` | Correct answers to the address orientation questions — set these to where the session is actually taking place. |
| `patient.name` | Spoken back to the participant in conversational transitions. Not scored. |
| `patient.dob` | Recorded with the session only; no question scores against it. |
| `assessor` | Written into the final report in `results/`. |
| `current_uk_pm`, `current_us_president` | Correct answers to two of the four Memory (Retrograde Memory — Famous People) items, 1 point each. A surname alone scores; a first name alone triggers a prompt for the surname; an incorrect full name scores 0. The other two items in that group (Thatcher, Kennedy) are fixed answers in `json/ACE-III.json`. |
| `previous_uk_pm`, `previous_us_president` | The outgoing leader. *"If there has been a recent change in leaders, probe for the name of the outgoing politician,"*

Day, date, month, year and season are taken from the system clock.

If `json/session_config.json` is missing, `get_session_config()` falls back to prompting for
each value on the command line at startup.

## How to Run

```bash
python main.py
```

### System Architecture and Workflow



### Directory Structure

```
ace/
├── main.py                     # entry point: builds state, runs graph, resumes checkpoints
├── graph.py                    # LangGraph state machine + per-question handlers
├── data_loader.py              # loads question bank, resolves DYNAMIC: answers
│
├── json/
│   ├── ACE-III.json            # question bank: domains, answers, score caps
│   └── session_config.json     # per-session values (see above)
│
├── LLM/
│   ├── LLM.py                  # local text model client
│   ├── vlm.py                  # local vision model client + response logging
│   └── dialogue.py             # intros, transitions, rephrasing, answer extraction
│
├── marking/
│   ├── marking.py              # deterministic scorers for every question type
│   └── preprocessing.py        # response normalisation (number words, ordinals)
│
├── visual_tasks/
│   ├── visual.py               # runs drawing tasks: prompts, timers, capture
│   ├── clock_scorer.py         # clock drawing (0-5)
│   ├── cube_scorer.py          # wire cube (0-2)
│   ├── infinity_scorer.py      # infinity diagram (0-1)
│   ├── writing.py              # sentence writing, checked via LanguageTool
│   └── pen_paper_scorer.py     # detects whether the participant is still drawing
│
├── camera/capture.py           # webcam stills/video with paper-edge detection
├── voice/
│   ├── capture.py              # microphone capture + faster-whisper transcription
│   └── tts.py                  # speech output through Furhat
├── virtual_avatar/avatar.py    # Furhat connection and gestures
├── ui/session_window.py        # Tkinter transcript window, saves progress on close
│
├── images/                     # ACE-III stimulus images (21)
│
├── test/
│   ├── marking_test.py         # scorer unit tests, run offline
│   ├── llm_test.py             # turn classification (needs LM Studio)
│   ├── vlm_test.py             # drawing scorers vs labelled image sets
│   ├── model_comparison.py     # benchmarks text/vision models
│   ├── vlm_confusion.py        # builds the confusion matrix figures
│   ├── Clocks/1..5/            # ground-truth drawings, folder name = expected score
│   ├── cube/1..2/
│   └── InfinitySymbol/0..1/
│
├── synthetic_personas/         # persona + transcript generation, replay, results
├── synthetic_transcripts/      # generated participant transcripts (200)
│
└── results/
    ├── ACE-III_*.json          # completed session reports
    ├── progress/               # mid-session checkpoints for resuming
    └── vlm_responses/          # raw vision model outputs
```


## Results

Each of the 200 synthetic participants was generated with a target cognitive status, run
through the full marking pipeline, and classified from its resulting ACE-III total by
`classify_total` in `synthetic_personas/results.py`: **>= 88 healthy, 77-87 MCI, < 77
dementia**. Rows are the designed status, columns the status the pipeline assigned.

| True \ Predicted | Healthy | MCI | Dementia | Total |
| --- | ---: | ---: | ---: | ---: |
| **Healthy** | **60** | 2 | 0 | 63 |
| **MCI** | 0 | **72** | 7 | 79 |
| **Dementia** | 0 | 0 | **58** | 58 |
| **Total** | 60 | 72 | 68 | 200 |

Overall agreement is **187/200 (93.5%)**. Per class: healthy 60/63 (95.2%), MCI 69/79
(87.3%), dementia 58/58 (100%).

All 13 disagreements sit on the severity-overestimating side — 3 healthy scored into the MCI
band and 10 MCI into the dementia band, with no case scored as less impaired than designed.
MCI is the weakest class, which is expected given it is the narrowest band at 11 points wide.

Reproduce with:

```bash
python -m synthetic_personas.results
```
