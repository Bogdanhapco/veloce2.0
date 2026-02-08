import streamlit as st
from huggingface_hub import InferenceClient
import tempfile
import os

st.title("Veloce 2.0")

# Load token from secrets
try:
    api_token = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("HF_TOKEN not found in secrets. Add it in .streamlit/secrets.toml (local) or app Secrets (deployed).")
    st.stop()

prompt = st.text_area("Enter your video prompt\n(e.g., 'A serene mountain lake at sunrise with birds flying and gentle wind sounds')", height=120)

# Aspect ratio selector (appended to prompt since direct control limited)
aspect_ratio = st.radio("Aspect Ratio", ["16:9 (Landscape)", "9:16 (Portrait)"], index=0)

# Duration selector (appended to prompt)
duration_sec = st.radio("Duration", ["10 seconds", "15 seconds"], index=0)

# Model selector (fallback options that work on Inference API)
model_options = {
    "Best Open-Source Quality (Wan2.2)": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Realistic Motion (Hunyuan)": "tencent/HunyuanVideo-1.5",
    "Fast & Reliable Fallback": "damo-vilab/text-to-video-ms-1.7b"
}
selected_model = st.selectbox("Model", list(model_options.keys()))
model_id = model_options[selected_model]

if st.button("Generate Video"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        # Enhance prompt with aspect/duration (since direct params not supported)
        enhanced_prompt = prompt
        if aspect_ratio == "16:9 (Landscape)":
            enhanced_prompt += ", widescreen 16:9 aspect ratio"
        else:
            enhanced_prompt += ", vertical 9:16 aspect ratio, portrait mode"
        enhanced_prompt += f", approximately {duration_sec} duration"

        with st.spinner("Generating video with synced audio... (1-10+ min; may queue)"):
            try:
                client = InferenceClient(token=api_token)

                # Core call — only prompt + model (no width/height/num_frames)
                video_bytes = client.text_to_video(
                    prompt=enhanced_prompt,
                    model=model_id,
                    # Optional: if a provider supports it, you can try parameters=dict(...)
                    # parameters={"num_inference_steps": 25, "guidance_scale": 7.0}  # Uncomment/test if errors allow
                )

                # Save temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_bytes)
                    video_path = tmp_file.name

                st.video(video_path)
                st.success("Video generated! (native audio where supported)")

                # Optional cleanup
                # os.remove(video_path)

            except Exception as e:
                st.error(f"Error: {str(e)}\n\n"
                         "Common fixes:\n"
                         "- Your token must have Inference API access (create new at huggingface.co/settings/tokens).\n"
                         "- Try a different model from the dropdown.\n"
                         "- Simplify prompt or use 10 seconds.\n"
                         "- Providers can queue heavy models — retry in a few minutes.\n"
                         "- For LTX-2 specifically: Not reliably on free Inference API yet; test in its official Space first (search 'LTX-2' on huggingface.co/spaces).")
