"""
coach10.py

Scenario:

You are an assistant coach of a basketball team. Your coach has tasked you
with creating more involved answers to questions that are time and environment
aware.

Using transformer agents, define tools that can be be called by the agent for
information that isn't part of original training.

Usage:

./.venv/bin/python ./coach10.py

"""

# FROM https://huggingface.co/docs/transformers/agents

from transformers import CodeAgent, HfEngine, Tool
import requests


class WeatherLookupTool(Tool):
    name = "get_the_weather"
    description = (
        "This is a tool that returns the weather of a specific location."
        "It returns the temperature in fahrenheit."
    )

    inputs = {
        "location": {
            "type": "text",
            "description": "the location (i.e. Dayton, Ohio)",
        }
    }
    output_type = "text"

    def forward(self, location: str) -> str:
        city = "Dayton"
        parts = list(filter(None, [x.strip() for x in location.split(",")]))
        if len(parts) >= 1:
            city = parts[0]

        geocode = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        ).json()

        latitude = 39.7589478
        longitude = -84.1916069
        if (
            len(geocode) > 0
            and "results" in geocode
            and len(geocode["results"]) > 0
            and "latitude" in geocode["results"][0]
            and "longitude" in geocode["results"][0]
        ):
            latitude = geocode["results"][0]["latitude"]
            longitude = geocode["results"][0]["longitude"]

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m&temperature_unit=fahrenheit&forecast_days=1"
        ).json()

        if "current" in weather and "temperature_2m" in weather["current"]:
            return f"{weather['current']['temperature_2m']}"

        return f"unknown"


llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")

agent = CodeAgent(
    tools=[WeatherLookupTool()], llm_engine=llm_engine, add_base_tools=False
)
response = agent.run("Can you give me the weather in Dayton, Ohio?")

print(f"Response: {response}")
