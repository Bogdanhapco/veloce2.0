import streamlit as st
from huggingface_hub import InferenceClient
import tempfile
import os
import asyncio
import edge_tts
from moviepy.editor import VideoFileClip, AudioFileClip

st.title("Veloce 2.0")

# Load your Hugging Face token from secrets (secure for deployment)
try:
    api_token = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("HF_TOKEN not found in secrets. Add it in .streamlit/secrets.toml (local) or in the app's Secrets settings (deployed on Streamlit Cloud). Get a free token at https://huggingface.co/settings/tokens")
    st.stop()

prompt = st.text_area(
    "Enter your video prompt (e.g., 'A cat jumping over the moon in a starry night sky')",
    height=120
)

aspect_ratio = st.radio("Aspect Ratio", ["16:9 (Landscape)", "9:16 (Portrait)"], index=0)
duration_sec = st.radio("Duration (approximate)", ["Short (~5-8 seconds)", "Longer (~10 seconds)"], index=0)

if st.button("Generate Video"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        # Enhance prompt since direct resolution/frames control is limited on this free model
        enhanced_prompt = prompt.strip()
        if aspect_ratio == "16:9 (Landscape)":
            enhanced_prompt += ", widescreen 16:9 aspect ratio, cinematic landscape view, horizontal format"
        else:
            enhanced_prompt += ", vertical 9:16 aspect ratio, portrait mode for mobile or social media, tall format"
        enhanced_prompt += f", smooth motion, approximately {duration_sec.lower()}, high detail, dynamic scene"

        with st.spinner("Generating AI video (silent clip) + adding voiceover... This free model may take 1-5 minutes"):
            try:
                client = InferenceClient(token=api_token)

                # Call the free text-to-video model (no credits needed for this one)
                video_bytes = client.text_to_video(
                    prompt=enhanced_prompt,
                    model="damo-vilab/text-to-video-ms-1.7b"
                )

                # Save silent video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                    tmp_vid.write(video_bytes)
                    video_path = tmp_vid.name

                # Generate free voiceover with EdgeTTS (Microsoft neural voices)
                async def generate_audio():
                    # Pick a voice (change if you want female/other accent)
                    voice = "en-US-GuyNeural"  # Good male US voice
                    # Limit text length to avoid TTS cut-off
                    tts_text = prompt[:300] + "..." if len(prompt) > 300 else prompt
                    communicate = edge_tts.Communicate(tts_text, voice)
                    audio_path = tempfile.mktemp(suffix=".mp3")
                    await communicate.save(audio_path)
                    return audio_path

                audio_path = asyncio.run(generate_audio())

                # Merge video + audio with MoviePy
                video_clip = VideoFileClip(video_path)
                audio_clip = AudioFileClip(audio_path)

                # Adjust audio to match video length (trim if longer, or you can loop if shorter)
                if audio_clip.duration > video_clip.duration:
                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                else:
                    audio_clip = audio_clip.set_duration(video_clip.duration)

                final_clip = video_clip.set_audio(audio_clip)
                final_path = tempfile.mktemp(suffix=".mp4")
                final_clip.write_videofile(final_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

                # Show the result
                st.video(final_path)
                st.success("Video generated! Short AI clip with synced voiceover (free model, quality is basic but no cost).")

                # Clean up temp files
                for path in [video_path, audio_path, final_path]:
                    if os.path.exists(path):
                        os.remove(path)

            except Exception as e:
                st.error(f"Oops — generation failed: {str(e)}\n\n"
                         "Common fixes:\n"
                         "- Make sure your HF_TOKEN is valid (free account/token works for this model).\n"
                         "- Try a shorter/simpler prompt.\n"
                         "- The free Inference API can queue or rate-limit — wait a minute and retry.\n"
                         "- If stuck, test the model directly in this free Space: https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis")
