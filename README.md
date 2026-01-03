# Video Analysis and Validation Agent

An analysis agent that evaluates whether a Hercules test run was executed as planned. It compares the agent's planning log, video evidence, and final test output to detect deviations.

## What It Does

1. **Parses the Planning Log** — Extracts the intended step-by-step actions from Hercules chat logs or JUnit XML.
2. **Inspects the Video(s)** — Samples frames and runs OCR to detect if each action is visibly executed.
3. **Cross-checks with Final Output** — Uses the test result to validate consistency.
4. **Produces a Deviation Report** — Flags each step as Observed, Skipped, Altered, Hallucinated, or Unclear.

## Requirements

- **Python 3.10+**
- **ffmpeg** (for video frame sampling)
- **tesseract** (for OCR)

Install system dependencies (macOS):
```bash
brew install ffmpeg tesseract
```

## Installation

```bash
# Clone and enter the directory
cd Video_analysis_and_validation_agent

# Create virtual environment and install dependencies
uv sync
```

## Usage

### Basic Usage (Auto-detects latest run)
```bash
uv run python analysis_agent.py
```

### Specify a Particular Run
```bash
uv run python analysis_agent.py \
  --scenario "Search_for_solid_blue_shirt,_verify_XL_size_availability,_add_to_cart_and_verify" \
  --run-id run_20260104_044235
```

### Skip Video Sampling (Screenshots Only)
```bash
uv run python analysis_agent.py --no-video-sampling
```

### Output as JSON
```bash
uv run python analysis_agent.py --output-format json --output opt/output/report.json
```

## Input Artifacts

The agent expects Hercules test artifacts in the following structure:
```
opt/
├── proofs/
│   └── <scenario_name>/
│       └── <run_id>/
│           ├── videos/
│           │   └── video_of_<scenario>.webm
│           └── screenshots/
│               ├── click_start_*.png
│               ├── click_end_*.png
│               └── ...
├── log_files/
│   └── <scenario_name>/
│       └── <run_id>/
│           └── log_between_*.json
└── output/
    └── <run_id>/
        └── *.xml (JUnit results)
```

## Output

Reports are written to `opt/output/deviation_report_<run_id>.md` by default.

### Sample Output
```
# Deviation Report

- Scenario: Search_for_solid_blue_shirt
- Run ID: run_20260104_044235
- Steps analyzed: 14
- Deviations: 5
- Average confidence: 72%

## Step Results
| Step | Description | Result | Conf | Notes |
| --- | --- | --- | --- | --- |
| 1 | Navigate to wrangler.in | Observed | 100% | Visual evidence found |
| 2 | Click on Search icon | Observed | 100% | Search overlay visible |
| 3 | Enter "solid blue shirt" | Deviation-Skipped | 100% | Text not visible in video |
| 4 | Press Enter | Deviation-Skipped | 100% | No expected text to validate |
```

## Deviation Taxonomy

Based on research from HALoGEN, BrowseComp, and HLE benchmarks:

| Status | Meaning |
|--------|---------|
| ✅ **Observed** | Clear visual evidence and log support found |
| 🟡 **Partially-Observed** | Only some expected signals found |
| ⏭️ **Deviation-Skipped** | Step was not executed (likely due to prior failure) |
| 🔀 **Deviation-Altered** | Executed differently than planned |
| 👻 **Hallucinated** | Logs claim success but no visual evidence |
| ❓ **Unclear** | Insufficient evidence to classify |

## Architecture

```
analysis_agent.py          # Main orchestrator
├── PlanParser             # Extracts steps from Hercules logs
├── VideoAnalyzer          # Frame extraction + OCR pipeline  
└── DeviationClassifier    # 6-class taxonomy classification

deviation_classifier.py    # Classification logic with confidence scoring
```

## Research Basis

This agent's design is informed by:
- **Fara-7B** — Structured action schemas for computer use
- **HALoGEN** — Hallucination detection methodology
- **BrowseComp** — Browsing agent evaluation criteria
- **DeepSeek OCR** — Vision pipeline for text extraction

## License

MIT
