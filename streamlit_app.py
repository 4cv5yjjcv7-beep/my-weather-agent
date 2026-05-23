import streamlit as st
import requests
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Streamlit UI Layout Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Wardrobe Advisor", page_icon="🧥")
st.title("🧥 Personal Wardrobe AI Agent")
st.caption("Powered by Gemini 1.5 Flash & Manual Execution")

# Initialize Chat Message History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Tell me where you are traveling, and I'll check the live weather to draft your custom clothing layout!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 2. Define the Agent Tools
# ---------------------------------------------------------------------------
def get_current_weather(city: str) -> str:
    """Fetches the current temperature and weather conditions for a given US city."""
    with st.status(f"🔧 Agent running NWS backend tool for '{city}'...", expanded=False) as status:
        try:
            clean_city = city.split(",")[0].strip()
            
            # 1. Geocoding (Open-Meteo is still the best free geocoder to get latitude/longitude)
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": clean_city, "count": 1, "language": "en", "format": "json"}
            
            geo_response = requests.get(geo_url, params=params)
            if geo_response.status_code != 200 or not geo_response.json().get("results"):
                status.update(label=f"❌ Location '{clean_city}' not found", state="error")
                return f"Could not find coordinates for {city}"
            
            # NWS prefers coordinates rounded to 4 decimal places
            lat = round(float(geo_response.json()["results"][0]["latitude"]), 4)
            lon = round(float(geo_response.json()["results"][0]["longitude"]), 4)
            
            # 2. National Weather Service (NWS) Points API
            headers = {'User-Agent': 'MyWeatherAgentApp/1.0 (your-email@example.com)'}
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            points_response = requests.get(points_url, headers=headers)
            
            if points_response.status_code != 200:
                status.update(label="❌ NWS Data unavailable (US only)", state="error")
                return f"Could not pull NWS weather for {city}. Is it outside the US?"
                
            points_data = points_response.json()
            forecast_url = points_data["properties"]["forecastHourly"]
            
            # 3. National Weather Service Hourly Forecast
            forecast_response = requests.get(forecast_url, headers=headers).json()
            
            # The 0th period is always the current, active hour
            current_hour = forecast_response["properties"]["periods"][0]
            
            temp = current_hour["temperature"]
            desc = current_hour["shortForecast"]

            status.update(label=f"✅ NWS Data fetched: {temp}°F, {desc}", state="complete")
            return f"Weather in {city}: {temp}°F, {desc}"
            
        except Exception as e:
            status.update(label="❌ Weather process crashed", state="error")
            return f"Weather lookup failed: {str(e)}"

TOOL_MAP = {"get_current_weather": get_current_weather}

# ---------------------------------------------------------------------------
# 3. Execution & Interactivity Loop
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Where are you going?")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        client = genai.Client()
        
        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a luxury personal wardrobe stylist agent. You MUST check the weather first using get_current_weather "
                "before finalizing your response. Provide an inspiring, structured clothing checklist based on the data."
            ),
            tools=[get_current_weather],
            temperature=0.3
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Build API history
            api_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "assistant" and "Hi! Tell me where" in msg["content"]:
                    continue
                api_role = "model" if msg["role"] == "assistant" else "user"
                api_history.append(
                    types.Content(role=api_role, parts=[types.Part.from_text(text=msg["content"])])
                )
                
            # Append the new prompt
            api_history.append(
                types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
            )
                
            with st.spinner("Stylist is consulting atmospheric records..."):
                import time
                
                # FIXED: Resilient helper function to gracefully survive 503 traffic spikes
                def call_gemini_with_retry(history_payload):
                    delay = 2
                    for attempt in range(3):
                        try:
                            return client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=history_payload,
                                config=config
                            )
                        except Exception as e:
                            # If the server is busy, show a toast, wait, and try again
                            if "503" in str(e) and attempt < 2:
                                st.toast(f"⚠️ Google servers busy. Retrying in {delay}s... (Attempt {attempt+1}/3)")
                                time.sleep(delay)
                                delay *= 2  # Double the wait time next time
                                continue
                            raise e

                # Send initial payload
                response = call_gemini_with_retry(api_history)
                
                # Manual Loop that forces the tracking 'id'
                while response.function_calls:
                    
                    # 1. Append the model's function request
                    api_history.append(response.candidates[0].content)
                    
                    # 2. Execute the tools
                    function_responses = []
                    for call in response.function_calls:
                        if call.name in TOOL_MAP:
                            call_args = dict(call.args) if call.args else {}
                            tool_result = TOOL_MAP[call.name](**call_args)
                            
                            kwargs = {
                                "name": call.name, 
                                "response": {"result": tool_result}
                            }
                            if hasattr(call, "id") and call.id:
                                kwargs["id"] = call.id
                                
                            function_responses.append(
                                types.Part.from_function_response(**kwargs)
                            )
                    
                    # 3. Append execution results back to history strictly as a 'user'
                    api_history.append(types.Content(role="user", parts=function_responses))
                    
                    # 4. Request the final synthesis from Gemini using our retry engine
                    response = call_gemini_with_retry(api_history)
                
            response_placeholder.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        # If the API crashes, this bypasses Streamlit's redaction filter to show us why!
        error_info = str(e)
        if hasattr(e, 'response_json') and e.response_json:
            error_info += f"\n\n**Raw Google Payload:**\n```json\n{e.response_json}\n```"
        st.error(f"### 🛑 Google API Crash\n{error_info}")