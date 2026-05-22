import streamlit as st
import requests
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Streamlit UI Layout Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Wardrobe Advisor", page_icon="🧥")
st.title("🧥 Personal Wardrobe AI Agent")
st.caption("Powered by Gemini 2.5 & Native Tool Calling")

# Initialize Chat Message History for the Streamlit UI display layer
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
    """Fetches the current temperature and weather conditions for a given city.
    
    Args:
        city: The name of the city to look up weather data for.
    """
    # Using a neat status box inside the tool run so users see execution live
    with st.status(f"🔧 Agent running backend tool for '{city}'...", expanded=False) as status:
        try:
            # Using OpenStreetMap's Nominatim API to find coordinates
            headers = {'User-Agent': 'MyWeatherAgentApp/1.0 (your-email@example.com)'}
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            
            geo_response = requests.get(geo_url, headers=headers)
            if geo_response.status_code != 200:
                status.update(label="❌ Geolocation connection failed", state="error")
                return {"error": "Failed to connect to geolocation service."}
                
            geo_res = geo_response.json()
            if not geo_res:
                status.update(label=f"❌ Location '{city}' not found", state="error")
                return {"error": f"Could not find coordinates for {city}"}
            
            lat = float(geo_res[0]["lat"])
            lon = float(geo_res[0]["lon"])
            
            # Open-Meteo weather request
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

            status.update(label=f"✅ Data fetched: {temp}°F, {desc}", state="complete")
            return {"city": city, "temperature_f": temp, "condition": desc}
            
        except Exception as e:
            status.update(label="❌ Weather process crashed", state="error")
            return {"error": f"Weather lookup failed: {str(e)}"}

# ---------------------------------------------------------------------------
# 3. Execution & Interactivity Loop
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Where are you going?")

if user_prompt:
    # Display user input in UI
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Initialize Gemini Client (pulls directly from Streamlit cloud secrets)
    client = genai.Client()
    
    # Configure tool configurations smoothly inside the Generative Configuration
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
        
        # 1. Rebuild the persistent API history structure
        api_history = []
        for msg in st.session_state.messages[:-1]:  # Exclude the prompt we just typed
            api_role = "model" if msg["role"] == "assistant" else "user"
            api_history.append(
                types.Content(role=api_role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
        with st.spinner("Stylist is consulting atmospheric records..."):
            # 2. Create an official stateful ChatSession using the SDK
            # The client.chats system handles internal tool payloads automatically!
            chat = client.chats.create(
                model="gemini-2.5-flash",
                config=config,
                history=api_history
            )
            
            # Send the text; the library automatically executes get_current_weather behind the scenes
            response = chat.send_message(user_prompt)
            
        # Display the final text output
        response_placeholder.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})