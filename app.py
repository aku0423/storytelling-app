import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
from gtts import gTTS
import io
import torch
import time


# -------------------------------------------------------------------
# 1. Image Captioning
# -------------------------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def img2text(image):
    processor, model = load_blip()
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# -------------------------------------------------------------------
# 2. Story Generation (optimised for length)
# -------------------------------------------------------------------
@st.cache_resource
def load_flan():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def text2story(caption):
    tokenizer, model = load_flan()
    prompt = (
        f"Write a short children's story of 50 to 100 words based on this description: {caption}. "
        f"The story should have a beginning, middle, and end. Use simple words for kids aged 3-10."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=280,
            min_length=80,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            num_beams=4,
            early_stopping=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.15
        )
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process: if too short, add a simple continuation
    if len(story.split()) < 40:
        story = story + " " + f"The {caption} was very happy. Everyone loved the story. The end."
    
    return story.strip()


# -------------------------------------------------------------------
# 3. Text-to-Speech (gTTS with retry + browser fallback)
# -------------------------------------------------------------------
def text2audio_gtss(story_text):
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            tts = gTTS(text=story_text, lang="en", slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise e
    return None

def text2audio_fallback(story_text):
    safe_text = story_text.replace("'", "\\'")
    js_code = f"""
        <script>
            (function() {{
                var utterance = new SpeechSynthesisUtterance('{safe_text}');
                utterance.lang = 'en-US';
                utterance.rate = 0.9;
                window.speechSynthesis.speak(utterance);
            }})();
        </script>
    """
    return js_code

def text2audio(story_text):
    try:
        audio_bytes = text2audio_gtss(story_text)
        return audio_bytes, None
    except Exception:
        fallback_html = text2audio_fallback(story_text)
        return None, fallback_html


# -------------------------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------------------------
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
            with st.spinner("Looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            with st.spinner("Writing a story..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                if word_count < 50:
                    st.warning("The story is a bit short. Let's imagine the rest!")
                elif word_count > 100:
                    st.info("The story is a bit long. Enjoy the extra magic!")

            with st.spinner("Preparing audio..."):
                audio_bytes, fallback_html = text2audio(story)
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("🎧 Listen to your story above!")
                else:
                    st.warning("Auto TTS unavailable – using browser speech.")
                    st.components.v1.html(fallback_html, height=0)
                    st.markdown(
                        f"""
                        <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance(`{story.replace('`', '\\`')}`))">
                            🔊 Play Story (Manual)
                        </button>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
