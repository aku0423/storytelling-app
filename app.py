"""
Storytelling Application for Kids
---------------------------------
- Uses BLIP directly (not pipeline) for robust image captioning.
- Generates a 50-100 word story with FLAN-T5.
- Converts story to audio with gTTS.
- Runs on Streamlit Cloud without task name errors.
"""

import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from gtts import gTTS
import io
import torch

# ------------------------------------------------------------
# 1. Load BLIP captioning model (direct, no pipeline)
# ------------------------------------------------------------
@st.cache_resource
def load_blip():
    """Load BLIP processor and model for image captioning."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def img2text(image):
    """
    Convert a PIL image to a text caption using BLIP.
    """
    processor, model = load_blip()
    # Prepare image (ensure RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ------------------------------------------------------------
# 2. Story generation (pipeline works fine for text)
# ------------------------------------------------------------
@st.cache_resource
def load_story_model():
    """Load FLAN-T5 for text generation."""
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1   # CPU
    )

def text2story(caption):
    """
    Expand a short caption into a full children's story (50-100 words).
    """
    story_pipeline = load_story_model()
    prompt = (
        f"Write a short children's story of 50 to 100 words based on this description: {caption}"
    )
    output = story_pipeline(prompt, max_length=150, do_sample=False)
    story = output[0]["generated_text"]
    return story.strip()

# ------------------------------------------------------------
# 3. Text-to-speech
# ------------------------------------------------------------
def text2audio(story_text):
    """
    Convert story text to audio (MP3) using Google Text-to-Speech.
    Returns an in-memory bytes buffer.
    """
    tts = gTTS(text=story_text, lang="en", slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# ------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Storytelling App for Kids", page_icon="📖")
    st.title("✨ Storytelling App ✨")
    st.markdown(
        "Upload an image, and I will create a magical 50–100 word children's story and read it aloud!"
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            # Step 1: Caption
            with st.spinner("Looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            # Step 2: Generate story
            with st.spinner("Writing a story just for you..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                if word_count < 50:
                    st.warning("The story is a bit short. Let's imagine the rest!")
                elif word_count > 100:
                    st.info("The story is a bit long. Enjoy the extra magic!")

            # Step 3: Convert to audio
            with st.spinner("Converting story to audio..."):
                audio_bytes = text2audio(story)
                st.audio(audio_bytes, format="audio/mp3")
                st.success("🎧 Listen to your story above!")

if __name__ == "__main__":
    main()
