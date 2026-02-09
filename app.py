"""
import streamlit as st
from gradio_client import Client
from gtts import gTTS
import time
from pathlib import Path

st.set_page_config(page_title="LTX-2 Video Generator", page_icon="🎥", layout="wide")

st.title("🎥 LTX-2 19B Video Generator")
st.markdown("Generate videos using LTX-2 19B Distilled model")

# Sidebar
st.sidebar.header("⚙️ Settings")

default_url = "https://5a84f7d44bed240468.gradio.live"
if "GRADIO_URL" in st.secrets:
    default_url = st.secrets["GRADIO_URL"]

gradio_url = st.sidebar.text_input("Gradio URL", value=default_url)

# Test connection
if st.sidebar.button("🔌 Test Connection"):
    try:
        client = Client(gradio_url)
        st.sidebar.success("✅ Connected!")
        with st.sidebar.expander("API Info"):
            st.code(str(client.view_api()))
    except Exception as e:
        st.sidebar.error(f"❌ {str(e)}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎬 Video Generation")
    
    prompt = st.text_area(
        "Prompt:",
        value="A cat playing piano in a cozy room, cinematic lighting, high quality",
        height=100,
        help="Describe the video you want to generate"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (optional):",
        value="blurry, low quality, distorted, ugly",
        height=60,
        help="What to avoid in the generation"
    )

with col2:
    st.subheader("⚙️ Parameters")
    
    # LTX-2 specific parameters
    num_frames = st.selectbox(
        "Frames",
        [49, 97, 121, 145],
        index=1,
        help="49=2s, 97=4s, 121=5s, 145=6s at 24fps"
    )
    
    width = st.selectbox("Width", [512, 704, 768, 1024], index=2)
    height = st.selectbox("Height", [512, 704, 768], index=1)
    
    guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 3.0, 0.5)
    num_steps = st.slider("Inference Steps", 10, 50, 30)
    
    use_seed = st.checkbox("Use fixed seed")
    if use_seed:
        seed = st.number_input("Seed", value=42, min_value=0)
    else:
        seed = -1

# Audio section
st.markdown("---")
add_audio = st.checkbox("🔊 Add audio narration")
if add_audio:
    audio_text = st.text_area("Narration text:", height=80)

# Generate button
st.markdown("---")

if st.button("🎬 Generate Video", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        try:
            with st.spinner("Connecting to LTX-2 model..."):
                client = Client(gradio_url)
            
            st.info("🎨 Generating video... This may take 2-5 minutes depending on length and GPU.")
            progress = st.progress(0)
            status = st.empty()
            
            # Try different LTX-2 parameter combinations
            result = None
            
            # Attempt 1: Full parameters with seed
            status.text("Attempting: Full parameters with seed...")
            progress.progress(10)
            try:
                result = client.predict(
                    prompt,
                    negative_prompt,
                    num_frames,
                    width,
                    height,
                    guidance_scale,
                    num_steps,
                    seed if use_seed else 42,
                    fn_index=0
                )
                status.text("✅ Success with full parameters!")
            except Exception as e1:
                # Attempt 2: Without seed
                status.text("Trying without seed parameter...")
                progress.progress(20)
                try:
                    result = client.predict(
                        prompt,
                        negative_prompt,
                        num_frames,
                        width,
                        height,
                        guidance_scale,
                        num_steps,
                        fn_index=0
                    )
                    status.text("✅ Success without seed!")
                except Exception as e2:
                    # Attempt 3: Just prompt and negative prompt
                    status.text("Trying minimal parameters...")
                    progress.progress(30)
                    try:
                        result = client.predict(
                            prompt,
                            negative_prompt,
                            fn_index=0
                        )
                        status.text("✅ Success with minimal params!")
                    except Exception as e3:
                        # Attempt 4: Just prompt
                        status.text("Trying just prompt...")
                        progress.progress(40)
                        try:
                            result = client.predict(prompt, fn_index=0)
                            status.text("✅ Success with prompt only!")
                        except Exception as e4:
                            # Attempt 5: Try fn_index=1
                            status.text("Trying alternate endpoint...")
                            try:
                                result = client.predict(prompt, fn_index=1)
                                status.text("✅ Success with fn_index=1!")
                            except Exception as e5:
                                st.error("❌ All attempts failed!")
                                with st.expander("Error Details"):
                                    st.code(f"1. Full params+seed: {e1}")
                                    st.code(f"2. Without seed: {e2}")
                                    st.code(f"3. Minimal: {e3}")
                                    st.code(f"4. Prompt only: {e4}")
                                    st.code(f"5. Alt endpoint: {e5}")
                                st.info("💡 Run `python ltx2_test.py` to diagnose the issue")
                                st.stop()
            
            if result:
                progress.progress(80)
                status.text("Processing output...")
                
                # Handle result
                if isinstance(result, str):
                    video_path = result
                elif isinstance(result, tuple):
                    video_path = result[0]
                else:
                    video_path = result
                
                progress.progress(100)
                status.empty()
                st.success("✅ Video generated successfully!")
                
                # Display
                st.video(video_path)
                
                # Download
                with open(video_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Video",
                        data=f,
                        file_name=f"ltx2_video_{int(time.time())}.mp4",
                        mime="video/mp4"
                    )
                
                # Audio
                if add_audio and audio_text.strip():
                    with st.spinner("Generating audio..."):
                        tts = gTTS(audio_text)
                        audio_path = output_dir / f"audio_{int(time.time())}.mp3"
                        tts.save(str(audio_path))
                        st.audio(str(audio_path))
                        
                        with open(audio_path, 'rb') as f:
                            st.download_button(
                                "📥 Download Audio",
                                data=f,
                                file_name=audio_path.name,
                                mime="audio/mp3"
                            )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Troubleshooting steps:")
            st.markdown("""
            1. Make sure Pinokio is running with LTX-2
            2. Test the Gradio URL in your browser
            3. Click 'Test Connection' in the sidebar
            4. Run `python ltx2_test.py` to find the correct parameters
            """)
            with st.expander("Debug Info"):
                st.code(str(e))

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <b>Tip:</b> Longer videos (145 frames) take more time and VRAM</p>
    <p>Powered by LTX-2 19B Distilled 🚀</p>
</div>
""", unsafe_allow_html=True)
