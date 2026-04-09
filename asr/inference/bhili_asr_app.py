import gradio as gr
import torch
import json
import os

NEMO_MODEL_PATH = "path/to/bhili_asr_finetune_v1_fixed.nemo"
TEST_MANIFEST_PATH = "path/to/test_manifest.jsonl"
TEST_RESULTS_PATH = "path/to/test_results.jsonl"

LANGUAGE_ID = "mr"

MODEL = None
TOP_SAMPLES = []


def load_model():
    global MODEL
    if MODEL is None:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = EncDecHybridRNNTCTCBPEModel.restore_from(
            NEMO_MODEL_PATH,
            map_location=device,
        )
        MODEL.eval()
        MODEL.freeze()
    return MODEL


def load_top_samples():
    global TOP_SAMPLES
    TOP_SAMPLES = []
    
    if os.path.exists(TEST_RESULTS_PATH):
        try:
            from jiwer import wer as compute_wer
            all_samples = []

            with open(TEST_RESULTS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        ref = item.get("reference", "").strip()
                        pred = item.get("prediction", "").strip()
                        audio_path = item.get("audio_filepath", "")
                        
                        if ref and pred and os.path.exists(audio_path):
                            try:
                                sample_wer = compute_wer(ref, pred)
                                all_samples.append({
                                    "audio_filepath": audio_path,
                                    "reference": ref,
                                    "wer": sample_wer
                                })
                            except:
                                continue

            all_samples.sort(key=lambda x: x["wer"])
            TOP_SAMPLES = all_samples[:10]
                
        except Exception:
            pass
    
    return TOP_SAMPLES


def get_sample_choices():
    if not TOP_SAMPLES:
        load_top_samples()
    
    choices = []
    for i in range(len(TOP_SAMPLES)):
        choices.append(f"sample-{i+1}.wav")
    
    return choices


def load_selected_sample(selection):
    if not selection:
        return None, ""
    
    try:
        sample_num = int(selection.split("-")[1].replace(".wav", "")) - 1
        
        if 0 <= sample_num < len(TOP_SAMPLES):
            sample = TOP_SAMPLES[sample_num]
            return sample["audio_filepath"], sample["reference"]
    except (IndexError, ValueError):
        pass
    
    return None, ""


def transcribe_audio(audio_path):
    if audio_path is None:
        return "Please upload an audio file or select a sample."

    if not os.path.exists(audio_path):
        return "Audio file not found."

    if os.path.getsize(audio_path) < 1000:
        return "Audio file appears too small."

    try:
        model = load_model()
        with torch.no_grad():
            result = model.transcribe(
                [audio_path],
                batch_size=1,
                language_id=LANGUAGE_ID,
            )

        if isinstance(result, tuple):
            result = result[0]

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if hasattr(result, "text"):
            result = result.text

        final_text = str(result).strip()
        return final_text if final_text else "No transcription generated."

    except Exception as e:
        return f"Error: {str(e)}"


def toggle_sample_section(visible):
    new_visible = not visible
    return (
        new_visible,
        gr.update(visible=new_visible),
        gr.update(visible=new_visible),
    )


def clear_all():
    return None, None, "", ""


load_top_samples()

custom_css = """
* {
    font-family: Inter, system-ui, -apple-system, sans-serif !important;
}

body, .gradio-container, .main, .wrap, .contain {
    background-color: #1a1a1a !important;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 2rem 4rem !important;
    color: #f5f5f5 !important;
    min-height: 100vh;
    background-color: #1a1a1a !important;
}

.main-header {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
}

.main-header h1 {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #ff7a18 !important;
    margin: 0 !important;
}

.main-header p {
    color: #888 !important;
    font-size: 0.95rem !important;
    margin-top: 0.3rem !important;
}

footer { display: none !important; }

button.primary {
    background-color: #ff7a18 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
}

button.primary:hover {
    background-color: #ff8c2a !important;
}

button.secondary {
    background-color: transparent !important;
    border: 1px solid #444 !important;
    color: #ccc !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover {
    background-color: #ff7a18 !important;
    border-color: #ff7a18 !important;
    color: #ffffff !important;
}

textarea, input, select {
    background-color: #252525 !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
}

label {
    color: #aaa !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

.gr-panel, .gr-box, .gr-form, .gr-block {
    background-color: #1a1a1a !important;
    border-color: #333 !important;
}

.gr-padded {
    background-color: #1a1a1a !important;
}
"""

with gr.Blocks(css=custom_css, title="Bhili ASR", theme=gr.themes.Base()) as demo:

    sample_visible = gr.State(False)

    gr.HTML("""
        <div class="main-header">
            <h1>Bhili ASR</h1>
            <p>Speech Recognition for Bhili Language</p>
        </div>
    """)

    sample_btn = gr.Button("Try with Sample", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(
                choices=get_sample_choices(),
                label="Select Sample",
                value=None,
                interactive=True,
                visible=False,
            )
            
            reference_display = gr.Textbox(
                label="Reference",
                interactive=False,
                lines=2,
                visible=False,
            )

            audio_input = gr.Audio(
                label="Audio",
                type="filepath",
                sources=["upload", "microphone"],
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Transcription",
                lines=6,
                interactive=False,
            )

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                transcribe_btn = gr.Button("Transcribe", variant="primary")

    sample_btn.click(
        fn=toggle_sample_section,
        inputs=[sample_visible],
        outputs=[sample_visible, sample_dropdown, reference_display],
    )

    sample_dropdown.change(
        fn=load_selected_sample,
        inputs=[sample_dropdown],
        outputs=[audio_input, reference_display],
    )
    
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[output_text],
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[audio_input, sample_dropdown, reference_display, output_text],
    )


if __name__ == "__main__":
    load_model()
    demo.launch(
        server_name="0.0.0.0",
        server_port=42069,
        share=True,
        show_error=True,
    )