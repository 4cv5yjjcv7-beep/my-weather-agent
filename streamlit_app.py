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
    """Fetches the current temperature and weather conditions for a given city."""
    with st.status(f"🔧 Agent running backend tool for '{city}'...", expanded=False) as status:
        try:
            headers = {'User-Agent': 'MyWeatherAgentApp/1.0 (your-email@example.com)'}
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            
            geo_response = requests.get(geo_url, headers=headers)
            if geo_response.status_code != 200:
                status.update(label="❌ Geolocation connection failed", state="error")
                return "Failed to connect to geolocation service."
                
            geo_res = geo_response.json()
            if not geo_res:
                status.update(label=f"❌ Location '{city}' not found", state="error")
                return f"Could not find coordinates for {city}"
            
            lat = float(geo_res[0]["lat"])
            lon = float(geo_res[0]["lon"])
            
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code&temperature_unit=fahrenheit"
            weather_res = requests.get(weather_url).json()
            
            temp = weather_res.get("current", {}).get("temperature_2m")
            code = weather_res.get("current", {}).get("weather_code", 0)
            
            desc = "Clear"
            if code in [1, 2, 3]: desc = "Partly Cloudy"
            elif code in [45, 48]: desc = "Foggy"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: desc = "Raining"
            elif code in [71, 73, 75, 77, 85, 86]: desc = "Snowing"
            elif code in [95, 96, 99]: desc = "Thunderstorm"

            status.update(label=f"✅ Data fetched: {temp}°F, {desc}", state="complete")
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
                
                # Send initial payload directly to the model
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=api_history,
                    config=config
                )
                
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
                            
                            # We safely pass call.id back to Google
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
                    
                    # 4. Request the final synthesis from Gemini
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=api_history,
                        config=config
                    )
                
            response_placeholder.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        # If the API crashes, this bypasses Streamlit's redaction filter to show us why!
        error_info = str(e)
        if hasattr(e, 'response_json') and e.response_json:
            error_info += f"\n\n**Raw Google Payload:**\n```json\n{e.response_json}\n```"
        st.error(f"### 🛑 Google API Crash\n{error_info}")