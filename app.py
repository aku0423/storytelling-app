import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
from gtts import gTTS
import io
import torch
import time


# -------------------------------------------------------------------
# 1. Image Captioning (BLIP)
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
# 2. Story Generation with FLAN-T5-Large (No Templates)
# -------------------------------------------------------------------
@st.cache_resource
def load_flan_large():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

def text2story(caption):
    """Generate a story using FLAN-T5-Large without any template fallback."""
    tokenizer, model = load_flan_large()
    
    # Instruction that forces a story output
    prompt = (
        f"Generate a children's story of 50-100 words based on this image description: '{caption}'. "
        f"The story must have a beginning, middle, and end. Use simple language. Start with 'Once upon a time'."
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            min_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_beams=4,
            early_stopping=True
        )
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Basic cleanup: remove any repetition of the prompt
    if story.startswith(prompt):
        story = story[len(prompt):]
    story = story.strip()
    
    # No template fallback – if empty, we return an error message (model-only)
    if len(story.split()) < 20:
        # Retry with different temperature
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.6,
            repetition_penalty=1.2,
            do_sample=True
        )
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if len(story.split()) < 20:
            story = "[Model failed to generate a story. Please try again with a different image.]"
    
    # Ensure length between 50-100 (trim only, no addition)
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100])
        if not story.endswith(('.', '!', '?')):
            story += '.'
    
    return story


# -------------------------------------------------------------------
# 3. Text-to-Speech
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
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise
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
        "Upload an image. AI describes it, then **generates an original children's story** using FLAN-T5-Large. "
        "No pre-written text – every story is created on the fly."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("📷 AI is looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            with st.spinner("📝 AI is writing a story (may take 20-30 seconds on first run)..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                if word_count < 50:
                    st.warning("The story is a bit short – the AI kept it concise.")
                elif word_count > 100:
                    st.info("A longer story – more fun for everyone!")

            with st.spinner("🔊 Converting story to audio..."):
                audio_bytes, fallback_html = text2audio(story)
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("🎧 Listen to your story above!")
                else:
                    st.warning("⚠️ Auto TTS unavailable – using browser voice.")
                    st.components.v1.html(fallback_html, height=0)
                    st.markdown(
                        f"""
                        <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance(`{story.replace('`', '\\`')}`))">
                            🔊 Play Story
                        </button>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
