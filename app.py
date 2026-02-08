import streamlit as st
from huggingface_hub import InferenceClient
import tempfile
import os
import asyncio
import edge_tts
from moviepy.editor import VideoFileClip, AudioFileClip

st.set_page_config(page_title="Veloce 2.0", layout="wide")

st.title("Veloce 2.0")
st.markdown("**Free AI text-to-video generator** with added voiceover — powered by Hugging Face (no payment required)")

# ── Load Hugging Face token from secrets ──
if "HF_TOKEN" not in st.secrets:
    st.error("HF_TOKEN is missing from secrets.\n\n"
             "**Local fix:** Create `.streamlit/secrets.toml` with:\n"
             "```toml\nHF_TOKEN = \"hf_xxxxxxxxxxxx\"\n```\n\n"
             "**Cloud fix:** Go to Manage app → Secrets → add HF_TOKEN")
    st.stop()

api_token = st.secrets["HF_TOKEN"]

# ── User inputs ──
prompt = st.text_area(
    "Describe your video",
    placeholder="A futuristic city at night with flying cars and neon lights, cinematic style",
    height=110
)

col1, col2 = st.columns(2)
with col1:
    aspect = st.radio("Aspect ratio", ["16:9 (widescreen)", "9:16 (vertical)"], index=0)
with col2:
    duration_choice = st.radio("Approximate length", ["Short (~5–8 s)", "Longer (~10 s)"], index=0)

if st.button("Generate Video", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please write a prompt first.")
        st.stop()

    # Build enhanced prompt (helps the model understand aspect & length)
    enhanced_prompt = prompt.strip()
    if aspect == "16:9 (widescreen)":
        enhanced_prompt += ", 16:9 aspect ratio, horizontal cinematic view"
    else:
        enhanced_prompt += ", 9:16 aspect ratio, vertical portrait format for mobile"
    enhanced_prompt += f", smooth animation, approximately {duration_choice.lower()}, detailed scene"

    with st.spinner("Generating video clip + voiceover (1–5 minutes)…"):
        try:
            client = InferenceClient(token=api_token)

            # ── Generate silent video (free model) ──
            video_bytes = client.text_to_video(
                prompt=enhanced_prompt,
                model="damo-vilab/text-to-video-ms-1.7b"
            )

            # Save video temporarily
            vid_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            vid_file.write(video_bytes)
            video_path = vid_file.name
            vid_file.close()

            # ── Generate voiceover (free EdgeTTS) ──
            async def create_voiceover():
                voice = "en-US-GuyNeural"          # change to en-GB-SoniaNeural, etc. if desired
                text = prompt[:280] + "…" if len(prompt) > 280 else prompt
                comm = edge_tts.Communicate(text, voice)
                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                await comm.save(audio_file.name)
                return audio_file.name

            audio_path = asyncio.run(create_voiceover())

            # ── Merge video + audio ──
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)

            # Match durations
            audio = audio.set_duration(video.duration)

            final = video.set_audio(audio)
            final_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            final.write_videofile(
                final_file.name,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None
            )
            final_path = final_file.name

            # ── Display result ──
            st.success("Done! Here's your video with voiceover.")
            st.video(final_path)

            # Clean up
            for p in [video_path, audio_path, final_path]:
                if os.path.exists(p):
                    os.remove(p)

        except Exception as e:
            st.error(f"Something went wrong:\n\n{str(e)}\n\n"
                     "**Quick fixes:**\n"
                     "• Check your HF_TOKEN is valid\n"
                     "• Try shorter prompt\n"
                     "• Wait 1–2 min and retry (free API can queue)\n"
                     "• Test model here: https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis")

st.markdown("---")
st.caption("Veloce 2.0 • Free Hugging Face model • Voice added with EdgeTTS • Deployed on Streamlit Cloud")
