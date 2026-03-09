"""
Tool definitions and implementations for the chatbot.

Each tool has two parts:
  1. A DEFINITION (dict) — sent to the OpenAI API so the LLM knows it exists.
  2. An IMPLEMENTATION (function) — called by the backend when the LLM uses it.

To add a new tool:
  - Add its definition to TOOL_DEFINITIONS
  - Add its implementation to TOOL_IMPLEMENTATIONS with the same name as key
"""

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from config import settings

logger = logging.getLogger(__name__)


# ── Tool Definitions (sent to OpenAI) ─────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get the current weather for a given city. "
                "Use this when the user asks about current weather, temperature, or conditions right now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris', 'New York, US'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default: celsius.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get a multi-day weather forecast for a given city (up to 5 days). "
                "Use this when the user asks about upcoming weather, tomorrow, this week, or a forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris', 'New York, US'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to forecast (1-5). Default: 3.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default: celsius.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": (
                "Get the current date and time. "
                "Use this when the user asks what time or date it is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone name, e.g. 'Europe/Paris', 'America/New_York'. "
                            "Default: UTC."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_file",
            "description": (
                "Generate and save a file with the given content. "
                "Use this when the user explicitly asks to create, generate, or export a file "
                "(e.g. 'create a CSV', 'generate a Python script', 'write a markdown report'). "
                "Supported formats: txt, md, csv, json, py, js, html, xml."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name with extension, e.g. 'cities.csv', 'script.py'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content as a string.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of what the file contains (shown to user).",
                    },
                },
                "required": ["filename", "content", "description"],
            },
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kelvin_to_celsius(k: float) -> float:
    return round(k - 273.15, 1)

def _kelvin_to_fahrenheit(k: float) -> float:
    return round((k - 273.15) * 9 / 5 + 32, 1)

def _geocode(location: str) -> tuple[float | None, float | None]:
    """Resolve a city name to lat/lon via OpenWeatherMap Geocoding API."""
    city = location.split(",")[0].strip()
    url = (
        f"http://api.openweathermap.org/geo/1.0/direct"
        f"?q={city}&limit=1&appid={settings.weather_api_key}"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, None
        return data[0]["lat"], data[0]["lon"]
    except Exception as exc:
        logger.error("Geocoding error for '%s': %s", location, exc)
        return None, None


# ── Tool Implementations ───────────────────────────────────────────────────────

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Fetch current weather from OpenWeatherMap and return a JSON string."""
    if not settings.weather_api_key:
        return json.dumps({"error": "WEATHER_API_KEY not configured."})

    lat, lon = _geocode(location)
    if lat is None:
        return json.dumps({"error": f"Could not find location: {location}"})

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={settings.weather_api_key}"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        convert = _kelvin_to_celsius if unit == "celsius" else _kelvin_to_fahrenheit
        unit_symbol = "°C" if unit == "celsius" else "°F"

        result = {
            "location": location,
            "datetime": datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M UTC"),
            "temperature": f"{convert(data['main']['temp'])}{unit_symbol}",
            "feels_like": f"{convert(data['main']['feels_like'])}{unit_symbol}",
            "humidity": f"{data['main']['humidity']}%",
            "description": data["weather"][0]["description"],
            "unit": unit,
        }
        logger.info("Weather fetched for '%s'", location)
        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        logger.error("Weather API error: %s", exc)
        return json.dumps({"error": str(exc)})


def get_weather_forecast(location: str, days: int = 3, unit: str = "celsius") -> str:
    """Fetch 5-day / 3-hour forecast from OpenWeatherMap and summarise by day."""
    if not settings.weather_api_key:
        return json.dumps({"error": "WEATHER_API_KEY not configured."})

    days = max(1, min(days, 5))
    lat, lon = _geocode(location)
    if lat is None:
        return json.dumps({"error": f"Could not find location: {location}"})

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={settings.weather_api_key}"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        convert = _kelvin_to_celsius if unit == "celsius" else _kelvin_to_fahrenheit
        unit_symbol = "°C" if unit == "celsius" else "°F"

        # Group 3-hour slots by day, compute min/max/dominant description
        by_day: dict[str, list] = defaultdict(list)
        for entry in data["list"]:
            day = entry["dt_txt"][:10]  # "YYYY-MM-DD"
            by_day[day].append(entry)

        forecast_days = []
        for day, entries in sorted(by_day.items())[:days]:
            temps = [e["main"]["temp"] for e in entries]
            descriptions = [e["weather"][0]["description"] for e in entries]
            dominant_desc = max(set(descriptions), key=descriptions.count)
            dt = datetime.strptime(day, "%Y-%m-%d")
            forecast_days.append({
                "date": dt.strftime("%A %d %B"),
                "min": f"{convert(min(temps))}{unit_symbol}",
                "max": f"{convert(max(temps))}{unit_symbol}",
                "description": dominant_desc,
            })

        result = {"location": location, "unit": unit, "forecast": forecast_days}
        logger.info("Forecast fetched for '%s': %d days", location, len(forecast_days))
        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        logger.error("Forecast API error: %s", exc)
        return json.dumps({"error": str(exc)})


def get_datetime(timezone: str = "UTC") -> str:
    """Return current date and time in the requested timezone."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        logger.warning("Unknown timezone '%s', falling back to UTC", timezone)
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    result = {
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%A %d %B %Y"),
        "time": now.strftime("%H:%M:%S"),
        "utc_offset": now.strftime("%z"),
    }
    logger.info("Datetime requested for timezone '%s'", timezone)
    return json.dumps(result, ensure_ascii=False)


ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".xml"}

def generate_file(filename: str, content: str, description: str) -> str:
    """Write content to a file and return a download URL."""
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return json.dumps({
            "error": f"Extension '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        })

    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = settings.files_dir / unique_name

    try:
        file_path.write_text(content, encoding="utf-8")
        logger.info("File generated: %s (%d chars)", file_path, len(content))
        return json.dumps({
            "success": True,
            "filename": safe_name,
            "stored_as": unique_name,
            "description": description,
            "download_url": f"/api/files/{unique_name}",
            "size_chars": len(content),
        })
    except Exception as exc:
        logger.error("File generation error: %s", exc)
        return json.dumps({"error": str(exc)})


# ── Dispatch table ─────────────────────────────────────────────────────────────

TOOL_IMPLEMENTATIONS: dict[str, callable] = {
    "get_current_weather":  lambda args: get_current_weather(**args),
    "get_weather_forecast": lambda args: get_weather_forecast(**args),
    "get_datetime":         lambda args: get_datetime(**args),
    "generate_file":        lambda args: generate_file(**args),
}


def execute_tool(name: str, arguments: str) -> str:
    """
    Parse the JSON arguments string from the LLM and call the matching function.
    Returns a JSON string result to feed back to the LLM.
    """
    if name not in TOOL_IMPLEMENTATIONS:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments)
        result = TOOL_IMPLEMENTATIONS[name](args)
        logger.info("Tool '%s' executed with args %s", name, args)
        return result
    except Exception as exc:
        logger.exception("Tool '%s' execution failed", name)
        return json.dumps({"error": str(exc)})