import streamlit as st
from gradio_client import Client
from gtts import gTTS
import os
import time
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎥",
    layout="wide"
)

# Title
st.title("🎥 Veloce")
st.markdown("Generate videos using Veloce model")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Gradio URL input
gradio_url = st.sidebar.text_input(
    "Gradio URL",
    value="https://5a84f7d44bed240468.gradio.live",
    help="The URL from your Pinokio Gradio instance (gradio.live link)"
)

# Test connection button
if st.sidebar.button("🔌 Test Connection"):
    try:
        client = Client(gradio_url)
        st.sidebar.success("✅ Connected successfully!")
        with st.sidebar.expander("Available API endpoints"):
            st.code(client.view_api())
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")
        st.sidebar.info("Make sure your Pinokio app is running!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Generation")
    
    # Video prompt
    prompt = st.text_area(
        "Enter your video prompt:",
        value="A cat playing piano in a cozy room, cinematic lighting",
        height=100,
        help="Describe the video you want to generate"
    )
    
    # Additional parameters
    with st.expander("🎛️ Advanced Settings"):
        duration = st.slider("Duration (frames)", 25, 121, 57, help="Number of frames to generate")
        width = st.selectbox("Width", [512, 704, 768], index=1)
        height = st.selectbox("Height", [512, 704, 768], index=1)
        guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 3.0, 0.5)
        num_inference_steps = st.slider("Inference Steps", 10, 50, 30)

with col2:
    st.subheader("Audio (Optional)")
    
    add_audio = st.checkbox("Add narration/audio")
    
    if add_audio:
        audio_text = st.text_area(
            "Narration text:",
            value="",
            height=100,
            help="Text to convert to speech"
        )
        
        voice_lang = st.selectbox(
            "Language",
            ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko"],
            help="Select TTS language"
        )

# Generate button
st.markdown("---")

if st.button("🎬 Generate Video", type="primary", use_container_width=True):
    
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Connect to Gradio
            with st.spinner("Connecting to LTX-2 model..."):
                client = Client(gradio_url)
            
            st.info("🎨 Generating video... This may take a few minutes.")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Call the Gradio API
            # Note: Adjust the parameters based on your actual Gradio interface
            # You may need to modify this based on the API structure
            status_text.text("Generating video...")
            progress_bar.progress(30)
            
            try:
                # Try common Gradio endpoints
                # Adjust based on your actual API (use client.view_api() to check)
                result = client.predict(
                    prompt,
                    duration,
                    width,
                    height,
                    guidance_scale,
                    num_inference_steps,
                    api_name="/generate"  # Change this if needed
                )
                
            except Exception as e:
                # Fallback to simpler call
                st.warning(f"Trying alternative API call... ({str(e)})")
                result = client.predict(
                    prompt,
                    api_name="/predict"
                )
            
            progress_bar.progress(80)
            status_text.text("Processing output...")
            
            # Display the video
            st.success("✅ Video generated successfully!")
            progress_bar.progress(100)
            
            # Handle different result types
            if isinstance(result, str):
                video_path = result
            elif isinstance(result, tuple):
                video_path = result[0]
            else:
                video_path = result
            
            # Display video
            st.video(video_path)
            
            # Download button
            with open(video_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Video",
                    data=f,
                    file_name=f"generated_video_{int(time.time())}.mp4",
                    mime="video/mp4"
                )
            
            # Generate audio if requested
            if add_audio and audio_text.strip():
                with st.spinner("Generating audio..."):
                    tts = gTTS(text=audio_text, lang=voice_lang)
                    audio_path = output_dir / f"audio_{int(time.time())}.mp3"
                    tts.save(str(audio_path))
                    
                    st.audio(str(audio_path))
                    st.info("💡 Tip: You can combine the video and audio using video editing software")
                    
                    with open(audio_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Audio",
                            data=f,
                            file_name=audio_path.name,
                            mime="audio/mp3"
                        )
            
        except Exception as e:
            st.error(f"❌ Error generating video: {str(e)}")
            st.info("💡 Troubleshooting tips:")
            st.markdown("""
            1. Make sure your Pinokio app is running
            2. Check that the Gradio URL is correct
            3. Click 'Test Connection' in the sidebar
            4. Check the API endpoints using the sidebar info
            """)
            
            with st.expander("🐛 Debug Info"):
                st.code(str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by LTX-2 running on your local GPU 🚀</p>
    <p>Make sure your Pinokio app is running before generating videos</p>
</div>
""", unsafe_allow_html=True)
