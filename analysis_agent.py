"""
Analysis Agent for Hercules Test Run Validation.

This agent evaluates whether a Hercules test run was executed as planned by comparing:
- The agent's planning log (thoughts/steps)
- The video evidence of the run
- The final test output

Based on research from Fara-7B, HALoGEN, and BrowseComp benchmarks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncio
from deviation_classifier import DeviationClassifier, StepStatus, ClassificationResult

# Optional: LLM-based vision analysis (superior to OCR)
try:
    from vision_llm_analyzer import VisionLLMAnalyzer
    VISION_LLM_AVAILABLE = True
except ImportError:
    VISION_LLM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Data Models (Fara-7B inspired structured schema)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """Structured plan step based on Fara-7B schema."""
    index: int
    action: str  # e.g., "navigate", "click", "type", "verify"
    target: str  # e.g., "Login button", "Search bar"
    description: str  # Full step description
    expected_visual_signal: str = ""  # What we expect to see on screen
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


@dataclass
class EvidenceItem:
    """A piece of evidence (screenshot or video frame)."""
    path: Path
    timestamp: float  # Seconds from start
    text: str = ""  # OCR extracted text
    source: str = "screenshot"  # "screenshot" or "video_frame"


@dataclass
class EvidenceStore:
    """Container for all evidence collected."""
    screenshots: list[EvidenceItem] = field(default_factory=list)
    video_frames: list[EvidenceItem] = field(default_factory=list)
    video_path: Optional[Path] = None


@dataclass
class StepResult:
    """Result of evaluating a single step."""
    index: int
    description: str
    status: str
    confidence: float
    notes: str = ""
    evidence_paths: list[str] = field(default_factory=list)
    timestamp_window: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Plan Parser
# ─────────────────────────────────────────────────────────────────────────────

class PlanParser:
    """
    Parses Hercules planning logs into structured PlanStep objects.
    Based on Agentic Organization principles for multi-step plan structuring.
    """

    def parse_from_chat_messages(self, chat_messages_path: Path) -> list[PlanStep]:
        """Extract plan steps from Hercules chat_messages.json or log files."""
        steps = []

        # Try to find planner agent messages
        if chat_messages_path.exists():
            with open(chat_messages_path) as f:
                data = json.load(f)

            # Look for user_proxy_agent messages containing plans
            for key, messages in data.items():
                for msg in messages:
                    if isinstance(msg.get("content"), dict):
                        plan_text = msg["content"].get("plan", "")
                        if plan_text:
                            steps = self._parse_plan_text(plan_text)
                            if steps:
                                return steps

        return steps

    def parse_from_junit_xml(self, junit_path: Path) -> list[PlanStep]:
        """Extract steps from JUnit XML test output."""
        steps = []
        if not junit_path.exists():
            return steps

        import xml.etree.ElementTree as ET
        tree = ET.parse(junit_path)
        root = tree.getroot()

        for i, testcase in enumerate(root.findall(".//testcase")):
            name = testcase.get("name", f"Step {i+1}")
            steps.append(PlanStep(
                index=i + 1,
                action="verify",
                target=name,
                description=name,
            ))

        return steps

    def parse_from_gherkin(self, gherkin_path: Path) -> list[PlanStep]:
        """
        Extract steps and visual signals from Gherkin feature files.
        Looks for # expected_visual_signals: ["signal1", "signal2"]
        """
        steps = []
        if not gherkin_path.exists():
            return steps

        content = gherkin_path.read_text()
        lines = content.splitlines()
        
        step_idx = 1
        current_step = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match Gherkin keywords
            if any(stripped.startswith(kw) for kw in ["Given ", "When ", "Then ", "And ", "But "]):
                description = re.sub(r"^(Given|When|Then|And|But)\s+", "", stripped)
                
                # Create new step
                current_step = PlanStep(
                    index=step_idx,
                    action=self._infer_action(description),
                    target=self._infer_target(description),
                    description=description,
                    expected_visual_signal="",
                )
                steps.append(current_step)
                step_idx += 1
            
            # Match expected signals in comments
            elif current_step and "# expected_visual_signals:" in stripped:
                try:
                    # Extract list from comment
                    signal_match = re.search(r"#\s*expected_visual_signals:\s*(\[.*?\])", stripped)
                    if signal_match:
                        signals = json.loads(signal_match.group(1).replace("'", '"'))
                        if isinstance(signals, list):
                            current_step.expected_visual_signal = ", ".join(signals)
                except Exception:
                    pass

        return steps

    def _parse_plan_text(self, plan_text: str) -> list[PlanStep]:
        """Parse numbered plan text into PlanStep objects."""
        steps = []
        lines = plan_text.strip().split("\n")

        for line in lines:
            # Match patterns like "1. Navigate to..." or "2. Click on..."
            match = re.match(r"^\s*(\d+)\.\s*(.+)$", line.strip())
            if match:
                idx = int(match.group(1))
                desc = match.group(2).strip()
                action = self._infer_action(desc)
                target = self._infer_target(desc)
                signal = self._infer_visual_signal(desc)

                steps.append(PlanStep(
                    index=idx,
                    action=action,
                    target=target,
                    description=desc,
                    expected_visual_signal=signal,
                ))

        return steps

    def _infer_action(self, description: str) -> str:
        """Infer action type from description."""
        desc_lower = description.lower()
        if "navigate" in desc_lower or "go to" in desc_lower or "open" in desc_lower:
            return "navigate"
        if "click" in desc_lower or "press" in desc_lower or "tap" in desc_lower:
            return "click"
        if "enter" in desc_lower or "type" in desc_lower or "input" in desc_lower:
            return "type"
        if "verify" in desc_lower or "check" in desc_lower or "assert" in desc_lower:
            return "verify"
        if "scroll" in desc_lower:
            return "scroll"
        if "wait" in desc_lower:
            return "wait"
        return "action"

    def _infer_target(self, description: str) -> str:
        """Extract target element from description."""
        # Look for quoted strings or common patterns
        quoted = re.findall(r'"([^"]+)"', description)
        if quoted:
            return quoted[0]
        # Look for "on the X" or "the X button/icon/field"
        match = re.search(r"(?:on|the)\s+(\w+(?:\s+\w+)?)\s+(?:button|icon|field|page|element)", description, re.I)
        if match:
            return match.group(1)
        return ""

    def _infer_visual_signal(self, description: str) -> str:
        """Infer expected visual signal from description."""
        desc_lower = description.lower()
        if "search" in desc_lower:
            return "search"
        if "login" in desc_lower:
            return "login"
        if "cart" in desc_lower:
            return "cart"
        if "product" in desc_lower:
            return "product"
        # Extract quoted text as expected signal
        quoted = re.findall(r'"([^"]+)"', description)
        if quoted:
            return quoted[0].lower()
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Video/Screenshot Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class VideoAnalyzer:
    """
    Analyzes video evidence using OCR (DeepSeek OCR inspired pipeline).
    """

    def __init__(self, video_interval: float = 2.0):
        self.video_interval = video_interval

    def extract_frames(self, video_path: Path, output_dir: Path) -> list[EvidenceItem]:
        """Extract frames from video at regular intervals using ffmpeg."""
        frames = []
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get video duration
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                    capture_output=True, text=True
                )
                duration_str = result.stdout.strip()
                if duration_str and duration_str != "N/A":
                    duration = float(duration_str)
                else:
                    duration = 60.0
            except ValueError:
                duration = 60.0

            # Extract frames
            subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-vf", f"fps=1/{self.video_interval}",
                 str(output_dir / "frame_%04d.png"), "-y", "-loglevel", "error"],
                check=True
            )

            # Collect frame paths with timestamps
            for i, frame_path in enumerate(sorted(output_dir.glob("frame_*.png"))):
                timestamp = i * self.video_interval
                frames.append(EvidenceItem(
                    path=frame_path,
                    timestamp=timestamp,
                    source="video_frame"
                ))

        except Exception as e:
            print(f"Warning: Could not extract frames: {e}")

        return frames

    def ocr_image(self, image_path: Path) -> str:
        """Run OCR on an image using tesseract."""
        try:
            result = subprocess.run(
                ["tesseract", str(image_path), "stdout", "-l", "eng"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def process_evidence(self, evidence: EvidenceStore) -> EvidenceStore:
        """Run OCR on all evidence items."""
        for item in evidence.screenshots + evidence.video_frames:
            if not item.text:
                item.text = self.ocr_image(item.path)
        return evidence


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Agent
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisAgent:
    """
    Main orchestrator for Hercules run validation.
    
    Implements the HALoGEN-inspired hallucination detection and 
    BrowseComp-inspired deviation classification.
    """

    def __init__(
        self,
        scenario: str,
        run_id: str,
        base_path: Path,
        video_interval: float = 2.0,
        no_video_sampling: bool = False,
        use_vision_llm: bool = False,  # Use LLM for vision analysis instead of OCR
        vision_provider: str = "groq",  # LLM provider for vision analysis
        gherkin_path: Path | None = None,  # Explicit path to Gherkin file
    ):
        self.scenario = scenario
        self.run_id = run_id
        self.base_path = base_path
        self.video_interval = video_interval
        self.no_video_sampling = no_video_sampling
        self.use_vision_llm = use_vision_llm and VISION_LLM_AVAILABLE
        self.gherkin_path = gherkin_path

        self.plan_parser = PlanParser()
        self.video_analyzer = VideoAnalyzer(video_interval)
        self.classifier = DeviationClassifier()
        
        # Initialize Vision LLM Analyzer if enabled
        if self.use_vision_llm:
            self.vision_llm = VisionLLMAnalyzer(provider=vision_provider)
            print(f"Vision LLM enabled using {vision_provider}")
        else:
            self.vision_llm = None

    def find_artifacts(self) -> dict:
        """Locate all artifacts for the run."""
        artifacts = {
            "video": None,
            "screenshots": [],
            "chat_messages": None,
            "junit_xml": None,
            "log_files": [],
            "gherkin": self.gherkin_path,
        }

        # First, try structured directories (opt/proofs, opt/log_files, etc.)
        # Proofs directory
        proofs_dir = self.base_path / "opt" / "proofs" / self.scenario / self.run_id
        if proofs_dir.exists():
            # Find video
            videos = list(proofs_dir.rglob("*.webm")) + list(proofs_dir.rglob("*.mp4"))
            if videos:
                artifacts["video"] = videos[0]

            # Find screenshots
            screenshots = list(proofs_dir.rglob("*.png"))
            artifacts["screenshots"] = screenshots

        # Log files directory
        log_dir = self.base_path / "opt" / "log_files" / self.scenario / self.run_id
        if log_dir.exists():
            # Find chat messages
            chat_files = list(log_dir.glob("*chat_manager*.json"))
            if chat_files:
                artifacts["chat_messages"] = chat_files[-1]  # Use latest

            artifacts["log_files"] = list(log_dir.glob("*.json"))

        # Output directory
        output_dir = self.base_path / "opt" / "output" / self.run_id
        if output_dir.exists():
            xml_files = list(output_dir.glob("*.xml"))
            if xml_files:
                artifacts["junit_xml"] = xml_files[0]

        # Search for Gherkin file
        if not artifacts["gherkin"]:
            gherkin_search_paths = [
                self.base_path / "gherkin_files" / f"{self.scenario}.feature",
                self.base_path / "opt" / "input" / f"{self.scenario}.feature",
            ]
            for path in gherkin_search_paths:
                if path.exists():
                    artifacts["gherkin"] = path
                    break

        # Fallback: Flat directory structure (e.g., supportingLogs)
        # This supports directories where all artifacts are in a single folder
        if not artifacts["video"]:
            videos = list(self.base_path.glob("*.webm")) + list(self.base_path.glob("*.mp4"))
            if videos:
                artifacts["video"] = videos[0]
        
        if not artifacts["screenshots"]:
            screenshots = list(self.base_path.glob("*.png"))
            artifacts["screenshots"] = screenshots
        
        if not artifacts["chat_messages"]:
            # Look for agent_inner_logs.json, chat_manager logs, or any logs
            agent_logs = list(self.base_path.glob("agent_inner_logs.json"))
            chat_logs = list(self.base_path.glob("*chat_manager*.json"))
            if chat_logs:
                artifacts["chat_messages"] = chat_logs[-1]
            elif agent_logs:
                artifacts["chat_messages"] = agent_logs[0]
            
            artifacts["log_files"] = list(self.base_path.glob("*.json"))
        
        if not artifacts["junit_xml"]:
            xml_files = list(self.base_path.glob("test_result.xml")) + list(self.base_path.glob("*.xml"))
            if xml_files:
                artifacts["junit_xml"] = xml_files[0]

        return artifacts

    def collect_evidence(self, artifacts: dict) -> EvidenceStore:
        """Collect and process all evidence."""
        evidence = EvidenceStore()

        # Collect screenshots
        for ss_path in artifacts.get("screenshots", []):
            # Parse timestamp from filename if possible
            timestamp = self._parse_timestamp_from_filename(ss_path.name)
            evidence.screenshots.append(EvidenceItem(
                path=ss_path,
                timestamp=timestamp,
                source="screenshot"
            ))

        # Extract video frames
        if artifacts.get("video") and not self.no_video_sampling:
            evidence.video_path = artifacts["video"]
            frames_dir = self.base_path / "opt" / "temp_frames" / self.run_id
            evidence.video_frames = self.video_analyzer.extract_frames(
                artifacts["video"], frames_dir
            )

        # Run OCR on all evidence
        evidence = self.video_analyzer.process_evidence(evidence)

        return evidence

    def _parse_timestamp_from_filename(self, filename: str) -> float:
        """Extract timestamp from screenshot filename."""
        # Pattern: action_end_1767481968094762000.png (nanoseconds)
        match = re.search(r"_(\d{19})\.png$", filename)
        if match:
            ns = int(match.group(1))
            return ns / 1e9  # Convert to seconds (relative, will normalize later)
        return 0.0

    def extract_plan(self, artifacts: dict) -> list[PlanStep]:
        """Extract plan steps from artifacts."""
        steps = []

        # Try Gherkin first (highest priority if available)
        if artifacts.get("gherkin"):
            steps = self.plan_parser.parse_from_gherkin(artifacts["gherkin"])

        # Try chat messages second
        if not steps and artifacts.get("chat_messages"):
            steps = self.plan_parser.parse_from_chat_messages(artifacts["chat_messages"])

        # Fallback to JUnit XML
        if not steps and artifacts.get("junit_xml"):
            steps = self.plan_parser.parse_from_junit_xml(artifacts["junit_xml"])

        return steps

    def align_evidence_to_steps(
        self, steps: list[PlanStep], evidence: EvidenceStore, artifacts: dict
    ) -> list[PlanStep]:
        """Align evidence timestamps to step windows."""
        if not steps:
            return steps

        # Parse action timestamps from screenshot filenames
        action_times = []
        for ss in evidence.screenshots:
            # Extract action type and timestamp
            match = re.match(r"(\w+)_(start|end)_(\d+)", ss.path.stem)
            if match:
                action_type = match.group(1)
                start_end = match.group(2)
                ns = int(match.group(3))
                action_times.append({
                    "action": action_type,
                    "type": start_end,
                    "timestamp": ns / 1e9,
                    "path": ss.path
                })

        # Try to use log files for timestamps if screenshots are insufficient
        log_files = artifacts.get("log_files", [])
        log_timestamps = []
        import datetime
        
        for log_file in log_files:
            # Filename format: log_between_sender-user-rec-chat_manager_YYYY-MM-DDTHH-MM-SS-mmmmm.json
            name = log_file.name
            if "log_between" in name and "chat_manager" in name:
                try:
                    # Extract timestamp part
                    ts_part = name.split("_")[-1].replace(".json", "")
                    # Parse simplified ISO format in filename
                    # 2026-01-04T19-13-06-569774 -> %Y-%m-%dT%H-%M-%S-%f
                    dt = datetime.datetime.strptime(ts_part, "%Y-%m-%dT%H-%M-%S-%f")
                    log_timestamps.append(dt.timestamp())
                except ValueError:
                    pass
        
        log_timestamps.sort()
        
        # If we have log timestamps, use them to approximate step starts
        if log_timestamps and len(log_timestamps) > 0:
            start_time_base = log_timestamps[0]
            
            # If we already have screenshot timestamps, align baselines
            video_start_offset = 0.0
            if action_times:
                # Assuming first screenshot generally happens shortly after first log
                # We can try to normalize. But simpler is to rely on relative times.
                pass
            
            # Assign windows based on log sequence (1 log file per interaction/step usually)
            # This is a heuristic: each log file corresponds to a step initiation
            
            # Map log timestamps to relative time from start
            relative_logs = [t - start_time_base for t in log_timestamps]
            
            # If we didn't get enough info from screenshots, overlay log info
            steps_with_time = [s for s in steps if s.timestamp_start > 0]
            if len(steps_with_time) < len(steps) * 0.5: # If < 50% steps have time from screenshots
                print(f"Using log timestamps for alignment (found {len(relative_logs)} logs)")
                
                # Ideally map 1 log -> 2 steps (Action + Validation) or 1 log -> 1 step logic
                # For now, let's distribute evenly or match indices
                for i, step in enumerate(steps):
                    # We typically have fewer logs than total steps because of the Action/Verify split
                    # Heuristic: Step N corresponds roughly to log index N // 2
                    log_idx = i // 2 
                    if log_idx < len(relative_logs):
                        step.timestamp_start = relative_logs[log_idx]
                        if log_idx + 1 < len(relative_logs):
                            step.timestamp_end = relative_logs[log_idx+1]
                        else:
                            step.timestamp_end = relative_logs[log_idx] + 10.0 # Default buffer

        return steps

    async def find_visual_matches(
        self, step: PlanStep, evidence: EvidenceStore
    ) -> list[EvidenceItem]:
        """Find evidence items that match the step's expected visual signal."""
        matches = []
        
        # Use Vision LLM for frame analysis if enabled
        if self.use_vision_llm and self.vision_llm and evidence.video_frames:
            # Sample frames to analyze (every 5th frame to reduce API calls)
            # Use the new async batch analyzer
            sampled_frames = evidence.video_frames[::5][:3]  # Max 3 frames
            
            expected_signals = []
            if step.expected_visual_signal:
                expected_signals.append(step.expected_visual_signal)
            if step.target:
                expected_signals.append(step.target)
            
            frame_data = [(f.path, f.timestamp) for f in sampled_frames]
            
            try:
                results = await self.vision_llm.analyze_frames_batch(
                    frames=frame_data,
                    expected_signals=expected_signals,
                    step_description=step.description,
                    sample_rate=1, # Already sampled
                )
                
                for i, result in enumerate(results):
                    if result.confidence > 0.5:
                        frame = sampled_frames[i]
                        frame.text = " ".join(result.key_text + result.visible_elements)
                        matches.append(frame)
            except Exception as e:
                print(f"Vision LLM error: {e}")
            
            return matches
        
        # Fallback to OCR-based matching
        signal = step.expected_visual_signal.lower() if step.expected_visual_signal else ""
        target = step.target.lower() if step.target else ""
        action = step.action.lower()

        search_terms = [s for s in [signal, target, action] if s]

        for item in evidence.screenshots + evidence.video_frames:
            text_lower = item.text.lower()
            # Check if any search term appears in OCR text
            for term in search_terms:
                if term and term in text_lower:
                    matches.append(item)
                    break

        return matches

    def check_log_claims(self, step: PlanStep, artifacts: dict) -> tuple[bool, bool]:
        """Check if logs claim this step was executed.
        
        Returns:
            Tuple of (has_claim, is_confirmed_success) where:
            - has_claim: True if logs mention this step being executed
            - is_confirmed_success: True if logs explicitly confirm success
        """
        chat_log = artifacts.get("chat_messages")
        if not chat_log:
            return False, False
        
        try:
            with open(chat_log) as f:
                data = json.load(f)
            
            # Extract keywords from step description for matching
            step_keywords = []
            desc_lower = step.description.lower()
            # Key action words
            for word in ["navigate", "click", "search", "enter", "type", "filter", 
                        "validate", "verify", "assert", "select", "locate",
                        "homepage", "load", "url", "open"]:
                if word in desc_lower:
                    step_keywords.append(word)
            # Key target words (URL, element names, etc)
            if step.target:
                step_keywords.append(step.target.lower())
            # Extract quoted values from description
            import re
            quoted = re.findall(r"'([^']+)'", step.description)
            step_keywords.extend([q.lower() for q in quoted])
            # Extract URL domains (e.g., wrangler.in from https://wrangler.in)
            urls = re.findall(r'https?://([^\s/.,]+(?:\.[^\s/.,]+)*)', step.description)
            step_keywords.extend([u.lower() for u in urls])
            
            # Handle agent_inner_logs.json format (planner_agent list)
            if "planner_agent" in data:
                messages = data["planner_agent"]
            else:
                # Flatten all message lists
                messages = []
                for key, msgs in data.items():
                    messages.extend(msgs)
            
            # Determine minimum matches required
            # Require at least 1 match, or 50% of keywords if more than 2
            min_matches = max(1, len(step_keywords) // 2) if step_keywords else 1
            
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    # Look for execution mention
                    if "previous_step:" in content_lower:
                        # Check if enough keywords match this content
                        matches = sum(1 for kw in step_keywords if kw in content_lower)
                        if matches >= min_matches:
                            is_success = "completed_successfully" in content_lower
                            return True, is_success
        except Exception:
            pass
        return False, False
    
    def detect_termination_step(self, artifacts: dict) -> int | None:
        """Detect which plan step caused test termination.
        
        Browser responses typically cover 2 plan steps (action + validation).
        We need to map browser response count to actual plan step numbers.
        """
        chat_log = artifacts.get("chat_messages")
        if not chat_log:
            return None
        
        try:
            with open(chat_log) as f:
                data = json.load(f)
            
            # Handle agent_inner_logs.json format (planner_agent list)
            if "planner_agent" in data:
                messages = data["planner_agent"]
                # Count completed step cycles (user msg from browser + planner response)
                # Each browser response contains "previous_step:" showing prior work
                browser_responses = 0
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, str) and "previous_step:" in content.lower():
                        browser_responses += 1
                    if isinstance(content, dict) and content.get("terminate") == "yes":
                        # Map browser responses to plan steps
                        # Typically: 1 response = 2 plan steps (action + validation)
                        # But the last response before termination covers only 1 step
                        # Pattern: 1×2=2, 2×2=4, 3×2=6, 4×2=8... but last is partial
                        # If terminated at response N, steps 1 to (N-1)*2 + 1 were attempted
                        # Skipped steps start from (N-1)*2 + 2
                        plan_step_at_failure = (browser_responses - 1) * 2 + 1
                        return plan_step_at_failure
            else:
                # Handle chat_manager format
                browser_responses = 0
                for key, messages in data.items():
                    for msg in messages:
                        content = msg.get("content", "")
                        if isinstance(content, str) and "previous_step:" in content.lower():
                            browser_responses += 1
                        if isinstance(content, dict) and content.get("terminate") == "yes":
                            plan_step_at_failure = (browser_responses - 1) * 2 + 1
                            return plan_step_at_failure
        except Exception:
            pass
        return None

    async def evaluate_steps(
        self, steps: list[PlanStep], evidence: EvidenceStore, artifacts: dict
    ) -> list[StepResult]:
        """Evaluate each step and classify deviations."""
        results = []
        previous_result = None
        
        # Detect at which step the test terminated
        termination_step = self.detect_termination_step(artifacts)
        test_terminated = False

        for step in steps:
            # If we've passed the termination point, all remaining steps are skipped
            if termination_step is not None and step.index > termination_step:
                test_terminated = True
            
            if test_terminated:
                # Mark as skipped due to early termination
                result = StepResult(
                    index=step.index,
                    description=step.description,
                    status=StepStatus.DEVIATION_SKIPPED,
                    confidence=0.95,
                    notes="Test terminated early; step was never executed.",
                    evidence_paths=[],
                    timestamp_window="",
                )
                results.append(result)
                previous_result = result
                continue
                
            # Find matching evidence
            visual_matches = await self.find_visual_matches(step, evidence)
            log_has_claim, log_confirmed_success = self.check_log_claims(step, artifacts)

            # Classify using the deviation classifier
            classification = self.classifier.classify(
                step=step,
                evidence=evidence,
                previous_result=previous_result,
                visual_matches=visual_matches,
                log_has_claim=log_has_claim,
                log_confirmed_success=log_confirmed_success,
            )

            # Build timestamp window string
            window_str = ""
            if step.timestamp_start or step.timestamp_end:
                window_str = f"{step.timestamp_start:.1f}-{step.timestamp_end:.1f}s"
                if step.action:
                    window_str += f" ({step.action})"

            result = StepResult(
                index=step.index,
                description=step.description,
                status=classification.status,
                confidence=classification.confidence,
                notes="; ".join(classification.notes),
                evidence_paths=[str(p) for p in classification.evidence_paths],
                timestamp_window=window_str,
            )

            results.append(result)
            previous_result = result

        return results

    def extract_final_output(self, artifacts: dict) -> dict:
        """Extract final test output from JUnit XML."""
        output = {"passed": False, "failure_message": ""}

        if artifacts.get("junit_xml"):
            import xml.etree.ElementTree as ET
            try:
                tree = ET.parse(artifacts["junit_xml"])
                root = tree.getroot()

                failures = root.findall(".//failure")
                if failures:
                    output["failure_message"] = failures[0].text or failures[0].get("message", "")
                else:
                    output["passed"] = True
            except Exception:
                pass

        # Also check chat messages for final response
        if artifacts.get("chat_messages"):
            try:
                with open(artifacts["chat_messages"]) as f:
                    data = json.load(f)
                for key, messages in data.items():
                    for msg in messages:
                        if isinstance(msg.get("content"), dict):
                            if msg["content"].get("terminate") == "yes":
                                output["failure_message"] = msg["content"].get("assert_summary", "")
                                output["passed"] = msg["content"].get("is_passed", False)
            except Exception:
                pass

        return output

    def generate_report(
        self,
        steps: list[PlanStep],
        results: list[StepResult],
        evidence: EvidenceStore,
        final_output: dict,
        output_path: Path,
        output_format: str = "markdown",
    ) -> None:
        """Generate the deviation report."""
        # Calculate summary stats
        deviation_count = sum(1 for r in results if "Deviation" in r.status or r.status == "Hallucinated")
        status_counts = {}
        for r in results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0

        if output_format == "json":
            report = {
                "scenario": self.scenario,
                "run_id": self.run_id,
                "steps_analyzed": len(results),
                "deviations": deviation_count,
                "average_confidence": avg_confidence,
                "status_counts": status_counts,
                "final_output": final_output,
                "results": [
                    {
                        "step": r.index,
                        "description": r.description,
                        "status": r.status,
                        "confidence": r.confidence,
                        "notes": r.notes,
                        "evidence": r.evidence_paths,
                        "timestamp_window": r.timestamp_window,
                    }
                    for r in results
                ],
            }
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        else:
            # Markdown report
            lines = [
                "# Deviation Report",
                "",
                f"- Scenario: {self.scenario}",
                f"- Run ID: {self.run_id}",
                f"- Steps analyzed: {len(results)}",
                f"- Deviations: {deviation_count}",
                f"- Average confidence: {avg_confidence*100:.1f}%",
                f"- Status counts: {', '.join(f'{k} {v}' for k, v in status_counts.items())}",
            ]

            if evidence.video_path:
                lines.append(f"- Proofs video: {evidence.video_path}")

            lines.append("")
            lines.append("## Final Output")
            if final_output.get("passed"):
                lines.append("- Result: ✅ Passed")
            else:
                lines.append(f"- Failure message: {final_output.get('failure_message', 'Unknown')}")

            lines.append("")
            lines.append("## Step Results")
            lines.append("| Step | Description | Result | Conf | Notes | Evidence |")
            lines.append("| --- | --- | --- | --- | --- | --- |")

            for r in results:
                evidence_str = ", ".join(r.evidence_paths[:2]) if r.evidence_paths else "-"
                notes = r.notes
                if r.timestamp_window:
                    notes = f"Window: {r.timestamp_window}; {notes}" if notes else f"Window: {r.timestamp_window}"
                lines.append(
                    f"| {r.index} | {r.description[:50]}{'...' if len(r.description) > 50 else ''} "
                    f"| {r.status} | {r.confidence*100:.0f}% | {notes} | {evidence_str} |"
                )

            lines.append("")

            with open(output_path, "w") as f:
                f.write("\n".join(lines))

        print(f"Report written to {output_path}")

    async def run(self, output_format: str = "markdown", output_path: Optional[Path] = None) -> None:
        """Execute the full analysis pipeline."""
        print(f"Analyzing run: {self.run_id} for scenario: {self.scenario}")

        # 1. Find artifacts
        artifacts = self.find_artifacts()
        print(f"Found artifacts: video={artifacts.get('video') is not None}, "
              f"screenshots={len(artifacts.get('screenshots', []))}, "
              f"logs={len(artifacts.get('log_files', []))}")

        # 2. Extract plan
        steps = self.extract_plan(artifacts)
        if not steps:
            print("Warning: Could not extract plan steps from artifacts.")
            # Create dummy steps from any available info
            steps = [PlanStep(index=1, action="unknown", target="", description="Unable to parse plan")]

        print(f"Extracted {len(steps)} plan steps")

        # 3. Collect evidence
        evidence = self.collect_evidence(artifacts)
        print(f"Collected evidence: {len(evidence.screenshots)} screenshots, "
              f"{len(evidence.video_frames)} video frames")

        # 4. Align evidence to steps
        steps = self.align_evidence_to_steps(steps, evidence, artifacts)

        # 5. Evaluate each step
        results = await self.evaluate_steps(steps, evidence, artifacts)

        # 6. Extract final output
        final_output = self.extract_final_output(artifacts)

        # 7. Generate report
        if output_path is None:
            output_path = self.base_path / "opt" / "output" / f"deviation_report_{self.run_id}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.generate_report(steps, results, evidence, final_output, output_path, output_format)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_run(base_path: Path) -> tuple[str, str]:
    """Find the latest run in the opt directory."""
    proofs_dir = base_path / "opt" / "proofs"
    if not proofs_dir.exists():
        raise FileNotFoundError("No proofs directory found")

    # Find all scenarios
    scenarios = [d for d in proofs_dir.iterdir() if d.is_dir()]
    if not scenarios:
        raise FileNotFoundError("No scenarios found in proofs directory")

    # Find latest run across all scenarios
    latest_run = None
    latest_scenario = None
    latest_time = None

    for scenario_dir in scenarios:
        for run_dir in scenario_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                # Parse timestamp from run_id
                try:
                    ts_str = run_dir.name.replace("run_", "")
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    if latest_time is None or ts > latest_time:
                        latest_time = ts
                        latest_run = run_dir.name
                        latest_scenario = scenario_dir.name
                except ValueError:
                    pass

    if latest_run is None:
        raise FileNotFoundError("No runs found")

    return latest_scenario, latest_run


def main():
    parser = argparse.ArgumentParser(
        description="Analysis Agent for Hercules Test Run Validation"
    )
    parser.add_argument("--scenario", type=str, help="Scenario name")
    parser.add_argument("--run-id", type=str, help="Run ID (e.g., run_20260104_043806)")
    parser.add_argument("--base-path", type=str, default=".",
                        help="Base path to the project directory")
    parser.add_argument("--output-format", choices=["markdown", "json"], default="markdown",
                        help="Output format for the report")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--video-interval", type=float, default=2.0,
                        help="Interval for video frame sampling (seconds)")
    parser.add_argument("--no-video-sampling", action="store_true",
                        help="Skip video frame extraction")
    parser.add_argument("--use-vision-llm", action="store_true",
                        help="Use LLM for vision analysis instead of OCR (more accurate)")
    parser.add_argument("--vision-provider", type=str, default="groq",
                        choices=["openai", "google", "anthropic", "groq"],
                        help="LLM provider for vision analysis")
    parser.add_argument("--gherkin", type=str, help="Explicit path to Gherkin feature file")

    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()

    # Auto-detect scenario and run if not provided
    scenario = args.scenario
    run_id = args.run_id

    if not scenario or not run_id:
        try:
            auto_scenario, auto_run = find_latest_run(base_path)
            scenario = scenario or auto_scenario
            run_id = run_id or auto_run
            print(f"Auto-detected: scenario={scenario}, run_id={run_id}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify --scenario and --run-id manually.")
            return

    # Create and run agent
    agent = AnalysisAgent(
        scenario=scenario,
        run_id=run_id,
        base_path=base_path,
        video_interval=args.video_interval,
        no_video_sampling=args.no_video_sampling,
        use_vision_llm=args.use_vision_llm,
        vision_provider=args.vision_provider,
        gherkin_path=Path(args.gherkin) if args.gherkin else None,
    )

    output_path = Path(args.output) if args.output else None
    asyncio.run(agent.run(output_format=args.output_format, output_path=output_path))


if __name__ == "__main__":
    main()
