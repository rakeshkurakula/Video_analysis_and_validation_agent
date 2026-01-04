"""
Vision LLM Analyzer for Video Frame Analysis

This module provides LLM-based vision analysis as a superior alternative to OCR.
It sends video frames as base64-encoded images to vision-capable LLMs (like GPT-4V, 
Gemini Vision, or Claude) for semantic understanding of UI state.

Research Foundation:
- HALoGEN: Detects hallucinations through multi-modal verification
- BrowseComp: Provides comprehensive browsing task evaluation
- DeepSeek-OCR: Inspiration for structured visual understanding
"""

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import dotenv

dotenv.load_dotenv()

@dataclass
class VisionAnalysisResult:
    """Result from LLM vision analysis of a frame."""
    frame_path: str
    timestamp: float
    visible_elements: list[str] = field(default_factory=list)
    page_title: str = ""
    url_visible: str = ""
    key_text: list[str] = field(default_factory=list)
    ui_state: str = ""  # e.g., "search_results", "homepage", "product_page"
    detected_actions: list[str] = field(default_factory=list)
    raw_response: str = ""
    confidence: float = 0.0


class VisionLLMAnalyzer:
    """
    Analyzes video frames using vision-capable LLMs instead of traditional OCR.
    
    Benefits over OCR:
    - Semantic understanding of UI state
    - Detection of visual elements (buttons, forms, products)
    - Understanding of page context and layout
    - Better handling of complex web designs
    
    Supported providers: openai, google, anthropic, groq
    """
    
    def __init__(
        self,
        provider: str = "groq",  # openai, google, anthropic, groq
        model: str = "openai/gpt-oss-120b",  # Default vision model
        api_key: str | None = None,
        base_url: str | None = None,
        config_path: Path | None = None,
    ):
        self.provider = provider.lower()
        self.config_path = config_path
        
        # Try to load from config file first
        config = self._load_config()
        if config:
            self.api_key = config.get("model_api_key", api_key or "")
            self.base_url = config.get("model_base_url", base_url or "")
            # For vision tasks, use vision-capable models (config model may not support images)
            if self.provider == "groq":
                self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Groq vision model (Llama 4)
            else:
                self.model = config.get("model_name", model).replace("openai/", "")
        else:
            self.api_key = api_key or self._get_api_key()
            self.base_url = base_url or ""
            # Default to vision-capable models
            if self.provider == "groq":
                self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
            else:
                self.model = model
    
    def _load_config(self) -> dict | None:
        """Load config from agents_llm_config.json if available."""
        config_paths = [
            self.config_path,
            Path("agents_llm_config.json"),
            Path(__file__).parent / "agents_llm_config.json",
        ]
        
        for path in config_paths:
            if path and path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    # Use planner_agent config for vision analysis
                    if self.provider in data:
                        return data[self.provider].get("planner_agent", {})
                except Exception:
                    pass
        return None
        
    def _get_api_key(self) -> str:
        """Get API key from environment or config."""
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "")
        elif self.provider == "google":
            return os.environ.get("GOOGLE_API_KEY", "")
        elif self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "")
        elif self.provider == "groq":
            return os.environ.get("GROQ_API_KEY", "")
        return ""
    
    def encode_image_base64(self, image_path: Path) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    async def analyze_frame(
        self,
        image_path: Path,
        expected_signals: list[str] | None = None,
        step_description: str = "",
        timestamp: float = 0.0,
    ) -> VisionAnalysisResult:
        """
        Analyze a single video frame using vision LLM.
        
        Args:
            image_path: Path to the frame image
            expected_signals: List of visual signals we expect to see
            step_description: Description of the step being verified
            timestamp: Timestamp of this frame in the video
            
        Returns:
            VisionAnalysisResult with detected elements and verification
        """
        if not self.api_key:
            return VisionAnalysisResult(
                frame_path=str(image_path),
                timestamp=timestamp,
                raw_response="Error: No API key configured",
            )
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(expected_signals, step_description)
        
        # Encode image
        image_b64 = self.encode_image_base64(image_path)
        
        # Call the appropriate API
        if self.provider == "openai":
            result = await self._call_openai(image_b64, prompt)
        elif self.provider == "google":
            result = await self._call_google(image_b64, prompt)
        elif self.provider == "anthropic":
            result = await self._call_anthropic(image_b64, prompt)
        elif self.provider == "groq":
            result = await self._call_groq(image_b64, prompt)
        else:
            result = {"error": f"Unknown provider: {self.provider}"}
        
        # Parse the response
        return self._parse_response(result, image_path, timestamp)
    
    def _build_analysis_prompt(
        self,
        expected_signals: list[str] | None,
        step_description: str,
    ) -> str:
        """Build the analysis prompt for the LLM."""
        prompt = """Analyze this screenshot from a web browser test execution.

Please identify and return the following in JSON format:
{
    "page_title": "The page title if visible",
    "url_visible": "Any URL visible in the address bar or on page",
    "visible_elements": ["List of key UI elements visible: buttons, forms, menus, etc."],
    "key_text": ["Important text content visible on the page"],
    "ui_state": "One of: homepage, search_results, product_page, checkout, form, error, loading, other",
    "detected_actions": ["Any user actions visible: clicked button, filled form, etc."]
}
"""
        if step_description:
            prompt += f"\n\nContext: We are verifying the step: '{step_description}'"
        
        if expected_signals:
            prompt += f"\n\nExpected visual signals to look for: {', '.join(expected_signals)}"
            prompt += "\n\nIn your response, also include:\n"
            prompt += '"signals_found": ["which of the expected signals are visible"],\n'
            prompt += '"signals_missing": ["which expected signals are NOT visible"]\n'
        
        return prompt
    
    async def _call_openai(self, image_b64: str, prompt: str) -> dict:
        """Call OpenAI's vision API."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
            )
            return {"content": response.choices[0].message.content}
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_google(self, image_b64: str, prompt: str) -> dict:
        """Call Google's Gemini Vision API."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            # Gemini expects different image format
            import io
            from PIL import Image
            
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            response = await model.generate_content_async([prompt, image])
            return {"content": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_anthropic(self, image_b64: str, prompt: str) -> dict:
        """Call Anthropic's Claude Vision API."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            return {"content": response.content[0].text}
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_groq(self, image_b64: str, prompt: str) -> dict:
        """Call Groq's vision API (OpenAI-compatible endpoint)."""
        try:
            import openai
            
            # Groq uses OpenAI-compatible API - ensure correct endpoint
            base_url = self.base_url or "https://api.groq.com"
            if not base_url.endswith("/openai/v1"):
                base_url = base_url.rstrip("/") + "/openai/v1"
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
            )
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
            )
            return {"content": response.choices[0].message.content}
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_response(
        self,
        response: dict,
        image_path: Path,
        timestamp: float,
    ) -> VisionAnalysisResult:
        """Parse the LLM response into a structured result."""
        result = VisionAnalysisResult(
            frame_path=str(image_path),
            timestamp=timestamp,
        )
        
        if "error" in response:
            result.raw_response = f"Error: {response['error']}"
            return result
        
        content = response.get("content", "")
        result.raw_response = content
        
        # Try to parse JSON from the response
        try:
            # Extract JSON from markdown code blocks if present
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            data = json.loads(content)
            result.page_title = data.get("page_title", "")
            result.url_visible = data.get("url_visible", "")
            result.visible_elements = data.get("visible_elements", [])
            result.key_text = data.get("key_text", [])
            result.ui_state = data.get("ui_state", "")
            result.detected_actions = data.get("detected_actions", [])
            result.confidence = 0.85  # High confidence for structured response
        except json.JSONDecodeError:
            # If JSON parsing fails, extract key information from text
            result.key_text = [content[:500]]  # Store raw response
            result.confidence = 0.5
        
        return result
    
    async def analyze_frames_batch(
        self,
        frames: list[tuple[Path, float]],  # (path, timestamp) pairs
        expected_signals: list[str] | None = None,
        step_description: str = "",
        sample_rate: int = 3,  # Analyze every Nth frame
    ) -> list[VisionAnalysisResult]:
        """
        Analyze multiple frames in parallel using asyncio.
        """
        import asyncio
        sampled_frames = frames[::sample_rate]
        
        tasks = []
        for path, timestamp in sampled_frames:
            tasks.append(self.analyze_frame(
                image_path=path,
                expected_signals=expected_signals,
                step_description=step_description,
                timestamp=timestamp,
            ))
        
        return await asyncio.gather(*tasks)
    
    async def verify_step(
        self,
        frames: list[tuple[Path, float]],
        step_description: str,
        expected_signals: list[str],
    ) -> dict:
        """
        Verify if a step was executed correctly based on video frames.
        """
        results = await self.analyze_frames_batch(
            frames=frames,
            expected_signals=expected_signals,
            step_description=step_description,
        )
        
        # Aggregate results
        all_signals_found = set()
        for r in results:
            all_signals_found.update(r.key_text)
            all_signals_found.update(r.visible_elements)
        
        # Check for expected signals
        signals_found = []
        signals_missing = []
        
        for signal in expected_signals:
            signal_lower = signal.lower()
            found = any(
                signal_lower in text.lower()
                for text in all_signals_found
            )
            if found:
                signals_found.append(signal)
            else:
                signals_missing.append(signal)
        
        verified = len(signals_missing) == 0
        confidence = len(signals_found) / len(expected_signals) if expected_signals else 0.5
        
        return {
            "verified": verified,
            "confidence": confidence,
            "signals_found": signals_found,
            "signals_missing": signals_missing,
            "evidence": results,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Example usage and CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision LLM Analyzer")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--signals", nargs="+", help="Expected visual signals")
    parser.add_argument("--step", default="", help="Step description")
    
    args = parser.parse_args()
    
    analyzer = VisionLLMAnalyzer(provider=args.provider, model=args.model)
    result = analyzer.analyze_frame(
        image_path=Path(args.image),
        expected_signals=args.signals,
        step_description=args.step,
    )
    
    print(json.dumps({
        "page_title": result.page_title,
        "url_visible": result.url_visible, 
        "visible_elements": result.visible_elements,
        "key_text": result.key_text,
        "ui_state": result.ui_state,
        "confidence": result.confidence,
    }, indent=2))
