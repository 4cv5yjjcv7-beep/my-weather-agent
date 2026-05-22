import os
import requests
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Streamlit UI Layout Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Wardrobe Advisor", page_icon="🧥")
st.title("🧥 Personal Wardrobe AI Agent")
st.caption("Powered by Gemini 2.5 & Autonomous Tool Calling")

# Initialize Chat Message History if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Tell me where you are traveling, and I'll check the live weather to draft your custom clothing layout!"}
    ]

# Display past chat history smoothly on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 2. Define the Agent Tools
# ---------------------------------------------------------------------------
def get_current_weather(city: str) -> dict:
    """Fetches the current temperature and weather conditions for a given city."""
    with st.spinner(f"🔧 Agent executing tool: Fetching live weather for {city}..."):
        try:
            # UPGRADE: Using OpenStreetMap's Nominatim API to find even the smallest towns
            headers = {'User-Agent': 'MyWeatherAgentApp/1.0 (your-email@example.com)'}
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            geo_res = requests.get(geo_url, headers=headers).json()
            
            if not geo_res:
                return {"error": f"Could not find coordinates for {city}"}
            
            # OpenStreetMap passes latitude and longitude back as strings, so we convert them to floats
            lat = float(geo_res[0]["lat"])
            lon = float(geo_res[0]["lon"])
            
            # The rest of your Open-Meteo weather request stays exactly the same!
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code&temperature_unit=fahrenheit"
            weather_res = requests.get(weather_url).json()
            
            current = weather_res.get("current", {})
            temp = current.get("temperature_2m")
            code = current.get("weather_code", 0)
            
            desc = "Clear"
            if code in [1, 2, 3]: desc = "Partly Cloudy"
            elif code in [45, 48]: desc = "Foggy"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: desc = "Raining"
            elif code in [71, 73, 75, 77, 85, 86]: desc = "Snowing"
            elif code in [95, 96, 99]: desc = "Thunderstorm"

            return {"city": city, "temperature_f": temp, "condition": desc}
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}

TOOL_MAP = {"get_current_weather": get_current_weather}

# ---------------------------------------------------------------------------
# 3. Autonomous Execution & Interactivity Loop
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Where are you going?")

if user_prompt:
    # Display user input in UI
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Initialize Gemini Client (will pull directly from Streamlit cloud secrets)
    client = genai.Client()
    
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a luxury personal wardrobe stylist agent. You MUST check the weather first using get_current_weather "
            "before finalizing your response. Provide an inspiring, structured clothing checklist based on the data."
        ),
        tools=list(TOOL_MAP.values()),
        temperature=0.3
    )

    with st.chat_message("assistant"):
        # Create a container so we can stream thoughts dynamically
        response_placeholder = st.empty()
        
        # Start chat chain
        chat = client.chats.create(model="gemini-2.5-flash", config=config)
        response = chat.send_message(user_prompt)
        
        # Run our autonomous tool calling execution engine loop
        while response.function_calls:
            function_responses = []
            for call in response.function_calls:
                if call.name in TOOL_MAP:
                    tool_result = TOOL_MAP[call.name](**call.args)
                    function_responses.append(
                        types.Part.from_function_response(name=call.name, response={"result": tool_result})
                    )
            response = chat.send_message(function_responses)
            
        # Display the final synthesized text result
        response_placeholder.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})