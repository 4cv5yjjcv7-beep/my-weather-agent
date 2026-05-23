import streamlit as st
import requests
import time
from datetime import datetime
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Streamlit UI Layout Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Wardrobe Advisor", page_icon="🧥")
st.title("🧥 Personal Wardrobe AI Agent")
st.caption("Powered by Gemini 2.5 Flash & NWS Forecasts")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Tell me your destination, your travel dates, and the occasion, and I'll pack your bags!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 2. Define the Agent Tools
# ---------------------------------------------------------------------------
# UPGRADE: We removed the 'days' limit. The tool now pulls the full 7-day calendar.
def get_trip_weather(city: str) -> str:
    """Fetches the 7-day weather forecast calendar for a given US city."""
    with st.status(f"🔧 Agent pulling 7-day forecast for '{city}'...", expanded=False) as status:
        try:
            clean_city = city.split(",")[0].strip()
            
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": clean_city, "count": 1, "language": "en", "format": "json"}
            
            geo_response = requests.get(geo_url, params=params)
            if geo_response.status_code != 200 or not geo_response.json().get("results"):
                status.update(label=f"❌ Location '{clean_city}' not found", state="error")
                return f"Could not find coordinates for {city}"
            
            lat = round(float(geo_response.json()["results"][0]["latitude"]), 4)
            lon = round(float(geo_response.json()["results"][0]["longitude"]), 4)
            
            headers = {'User-Agent': 'MyWeatherAgentApp/1.0'}
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            points_response = requests.get(points_url, headers=headers)
            
            if points_response.status_code != 200:
                status.update(label="❌ NWS Data unavailable", state="error")
                return f"Could not pull NWS weather for {city}. Is it outside the US?"
                
            points_data = points_response.json()
            forecast_url = points_data["properties"]["forecast"]
            forecast_response = requests.get(forecast_url, headers=headers).json()
            
            periods = forecast_response["properties"]["periods"]
            
            forecast_summary = []
            for p in periods:
                # UPGRADE: We explicitly attach the calendar date to the forecast so the AI can map it!
                calendar_date = p['startTime'][:10]
                forecast_summary.append(f"{p['name']} ({calendar_date}): {p['temperature']}°F, {p['shortForecast']}")
                
            result_string = f"7-Day Forecast for {city}:\n" + "\n".join(forecast_summary)
            status.update(label=f"✅ 7-Day NWS Calendar fetched", state="complete")
            return result_string
            
        except Exception as e:
            status.update(label="❌ Weather process crashed", state="error")
            return f"Weather lookup failed: {str(e)}"

TOOL_MAP = {"get_trip_weather": get_trip_weather}

# ---------------------------------------------------------------------------
# 3. Execution & Interactivity Loop
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("E.g., I'm going to Miami next weekend for a beach wedding...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        client = genai.Client()
        
        # UPGRADE: Tell the AI the current date so it understands "next weekend" or "tomorrow"
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        config = types.GenerateContentConfig(
            system_instruction=(
                f"Today is {current_date}. You are a luxury personal wardrobe stylist agent. "
                "You MUST use the get_trip_weather tool to pull the 7-day forecast for the destination. "
                "Filter the returned 7-day calendar data down to the specific dates or date range the user requested. "
                "If their dates are beyond the 7-day forecast window, kindly inform them you can only see 7 days out. "
                "Otherwise, analyze their occasion, the duration, and the specific weather for those dates to "
                "provide an inspiring, day-by-day structured clothing and packing itinerary."
            ),
            tools=[get_trip_weather],
            temperature=0.3
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            api_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "assistant" and "Hi! Tell me your destination" in msg["content"]:
                    continue
                api_role = "model" if msg["role"] == "assistant" else "user"
                api_history.append(
                    types.Content(role=api_role, parts=[types.Part.from_text(text=msg["content"])])
                )
                
            api_history.append(
                types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
            )
                
            with st.spinner("Stylist is mapping out your daily itinerary..."):
                # UPDATED: Handles both 503 server overloads AND 429 rate limits automatically
                def call_gemini_with_retry(history_payload):
                    delay = 2
                    for attempt in range(4): # Increased to 4 attempts to allow room for a rate-limit cooldown
                        try:
                            return client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=history_payload,
                                config=config
                            )
                        except Exception as e:
                            err_msg = str(e)
                            
                            # Case A: If we hit a 429 Rate Limit, pause for 14 seconds and try again
                            if "429" in err_msg and attempt < 3:
                                st.toast("⏳ Rate limit hit! Pausing 14s for free-tier cooldown...")
                                time.sleep(14)
                                continue
                                
                            # Case B: If we hit a 503 Traffic Spike, use exponential backoff
                            if "503" in err_msg and attempt < 3:
                                st.toast(f"⚠️ Google servers busy. Retrying in {delay}s... (Attempt {attempt+1}/4)")
                                time.sleep(delay)
                                delay *= 2
                                continue
                                
                            # If it's an unrelated error, raise it immediately
                            raise e
                
                while response.function_calls:
                    api_history.append(response.candidates[0].content)
                    
                    function_responses = []
                    for call in response.function_calls:
                        if call.name in TOOL_MAP:
                            call_args = dict(call.args) if call.args else {}
                            tool_result = TOOL_MAP[call.name](**call_args)
                            
                            kwargs = {"name": call.name, "response": {"result": tool_result}}
                            if hasattr(call, "id") and call.id:
                                kwargs["id"] = call.id
                                
                            function_responses.append(types.Part.from_function_response(**kwargs))
                    
                    api_history.append(types.Content(role="user", parts=function_responses))
                    response = call_gemini_with_retry(api_history)
                
            response_placeholder.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        error_info = str(e)
        if hasattr(e, 'response_json') and e.response_json:
            error_info += f"\n\n**Raw Google Payload:**\n```json\n{e.response_json}\n```"
        st.error(f"### 🛑 Google API Crash\n{error_info}")