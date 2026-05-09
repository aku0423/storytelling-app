import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from gtts import gTTS
import io
import torch
import time
import re


# -------------------------------------------------------------------
# 1. Image Captioning (BLIP from Hugging Face)
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
# 2. Story Generation with GPT-2 – No Templates
# -------------------------------------------------------------------
@st.cache_resource
def load_gpt2():
    """Load GPT-2 text generation pipeline."""
    return pipeline(
        "text-generation",
        model="gpt2",
        device=-1  # CPU
    )

def remove_repetitive_cycles(text):
    """Remove repetitive n-gram cycles (e.g., 'dog ran. dog ran. dog ran.') algorithmically."""
    words = text.split()
    if len(words) < 10:
        return text
    # Look for repeating 3-gram cycles
    for n in range(3, 7):
        seen = {}
        for i in range(len(words) - n):
            gram = tuple(words[i:i+n])
            if gram in seen:
                cycle_start = seen[gram]
                cycle_end = i
                # If the cycle repeats more than twice, cut after first occurrence
                if (cycle_end - cycle_start) > 0 and (cycle_end + (cycle_end - cycle_start)) < len(words):
                    if words[cycle_start:cycle_end] == words[cycle_end:cycle_end + (cycle_end - cycle_start)]:
                        words = words[:cycle_end]
                        break
        else:
            continue
        break
    return ' '.join(words)

def generate_story(caption, generator, temperature=0.7, repetition_penalty=1.2):
    """Generate a story using GPT-2 with no hardcoded templates."""
    prompt = f"Once upon a time, {caption}. "
    output = generator(
        prompt,
        max_new_tokens=130,
        min_new_tokens=50,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        truncation=True,
        pad_token_id=50256
    )
    full_text = output[0]['generated_text']
    # Remove the prompt
    story = full_text[len(prompt):].strip()
    # Basic cleanup: remove leading nonsense
    story = re.sub(r'^[^a-zA-Z]+', '', story)
    # Remove repetitive cycles
    story = remove_repetitive_cycles(story)
    # Ensure it ends with a sentence terminator
    if story and story[-1] not in '.!?':
        story += '.'
    return story

def text2story(caption):
    """Main story function: pure model generation with fallback retries (still model-only)."""
    generator = load_gpt2()
    
    # First attempt
    story = generate_story(caption, generator, temperature=0.7, repetition_penalty=1.2)
    word_count = len(story.split())
    
    # If too short (<30 words) or seems repetitive, try again with lower temperature
    if word_count < 30 or len(set(story.split())) < word_count * 0.6:
        story = generate_story(caption, generator, temperature=0.6, repetition_penalty=1.3)
        word_count = len(story.split())
    
    # If still too short, try once more with a different prompt variation (still no template)
    if word_count < 30:
        prompt2 = f"{caption} One day, "
        output = generator(
            prompt2,
            max_new_tokens=100,
            min_new_tokens=40,
            do_sample=True,
            temperature=0.65,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3
        )
        story = output[0]['generated_text'][len(prompt2):].strip()
        story = remove_repetitive_cycles(story)
        if story and story[-1] not in '.!?':
            story += '.'
        word_count = len(story.split())
    
    # Final length adjustment (only trimming, no template addition)
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100])
        if not story.endswith(('.', '!', '?')):
            story += '.'
    
    return story


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
        "Upload an image. AI describes it, then **writes an original story** (50–100 words) using GPT-2. No pre‑written templates – every story is generated fresh."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("📷 AI is looking at the image..."):
                caption = img2text(image)
                st.info(f"📷 *What I see:* {caption}")

            with st.spinner("📝 AI is writing a story (this may take 10-15 seconds)..."):
                story = text2story(caption)
                word_count = len(story.split())
                st.success(f"📖 *Your Story* ({word_count} words):")
                st.write(story)

                if word_count < 50:
                    st.warning("The story is a bit short – the AI generated a concise tale.")
                elif word_count > 100:
                    st.info("A longer story – enjoy the extra magic!")

            with st.spinner("🔊 Converting story to audio..."):
                audio_bytes, fallback_html = text2audio(story)
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("🎧 Listen to your story above!")
                else:
                    st.warning("⚠️ Auto TTS unavailable – using your browser's voice instead.")
                    st.components.v1.html(fallback_html, height=0)
                    st.markdown(
                        f"""
                        <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance(`{story.replace('`', '\\`')}`))">
                            🔊 Click to Play Story
                        </button>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
