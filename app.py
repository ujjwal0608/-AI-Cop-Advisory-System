import gradio as gr
from PIL import Image
import torch
import asyncio
import sys

import tomato_backend as backend

# The fix is now applied directly in tomato_backend.py, so this redefinition is no longer strictly necessary
backend.transformation = backend.transformation

model_engine = backend.load_prediction_engine()
gemini_client = backend.get_getni_client() if hasattr(backend, 'get_getni_client') else backend.get_gemni_client()

def plant_diagnostic_app(input_image):
    if input_image is None:
        return "⚠️ Please upload an image profile to begin", ""

    image_tensor = backend.transformation(input_image.convert("RGB")).unsqueeze(0).to(backend.device)

    with torch.no_grad():
        outputs = model_engine(image_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_idx = preds[0].item()

    raw_label = backend.dataset_classes[predicted_idx]
    clean_label = raw_label.replace("_", " ").replace("  ", " ").strip()

    if "healthy" in clean_label.lower():
        return "🩺 Diagnosis: Perfectly Healthy Leaf Profile 🌱", "🌱 **Crop Status: Normal**\n\nThe image sample metrics are sound. Keep up with your standard crop rotation sequences, hydration parameters, and soil balancing routines."

    prompt = f"""
    You are an expert plant pathologist and agricultural advisor.
    A tomato plant leaf sample has been diagnosed with the following pathology: {clean_label}.

    Provide a concise, highly professional agricultural field advisory using exactly these markdown headings:
    ### 🔬 Disease Analysis
    ### ⚠️ Primary Environmental Root Causes
    ### 🛡️ Preventative Field Measures (Max 3 clear points)
    ### 🧪 Prescriptive Treatment Options (Provide organic vs chemical alternatives)
    """

    gr.Info('Consulting Gemini Expert Plant Pathologist....')

    try:
        # ✅ FIXED: Changed .model to .models
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        ai_insights = response.text
    except Exception as e:
        ai_insights = f"⚠️ **Advisory API Connection Timeout:** Unable to extract dynamic crop logs. System Data: {str(e)}"

    return f"🩺 Diagnosis: {clean_label.title()}", ai_insights

agritech_theme = gr.themes.Default(
    primary_hue = 'emerald',
    secondary_hue = 'slate',
    neutral_hue = 'slate'
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_label_text_color="*primary_400",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500"
)
custom_css = """
app-header { text-align: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #334155; }
.output-banner { background: #0f172a !important; border: 1px solid #10b981 !important; border-radius: 8px !important; padding: 15px !important; font-size: 1.15rem !important; font-weight: bold !important;}
.advise-area { font-family: 'Inter', sans-serif; line-height: 1.7; padding: 10px; }
"""

with gr.Blocks(theme = agritech_theme, css = custom_css, title = "Tomato Leaf Health AI Dashboard") as demo:

    with gr.Group(elem_classes = 'app-header'):
        gr.Markdown("# 🍅 Tomato Plant Disease Diagnostics & AI Advisory")
        gr.Markdown('🧪 Dual-engine diagnostic application: Deep Residual Feature Maps (**ResNet-18**) fused with Generative Pathological Advisory (**Gemini 2.5 Flash**).')

    with gr.Row():
        with gr.Column(scale = 1):
            gr.Markdown('### 📤 Crop Sample Ingestion')
            image_input = gr.Image(type="pil", label="Target Leaf Photograph", show_label=True)
            submit_btn = gr.Button("Execute Diagnostic Analysis Scan", variant="primary", size="lg")

        with gr.Column(scale = 1):
            gr.Markdown("### 🔬 Precision System Analysis")
            label_output = gr.Textbox(label="Diagnostic Classification Signature", interactive=False, placeholder="Awaiting pipeline activation...", elem_classes="output-banner")
            markdown_output = gr.Markdown("### 📋 Expert Advisory Field Report\n*Drop an image file profile on the intake panel and trigger the diagnostic scan to map automated prescriptive crop remediation steps directly from Gemini.*", elem_classes="advise-area")

    submit_btn.click(
        fn = plant_diagnostic_app,
        inputs = image_input,
        outputs = [label_output, markdown_output]
    )

# ✅ FIXED: Safe main runtime guarding and Event Loop clean room isolation
if __name__ == "__main__":
    try:
        # Close any lingering event loop from previous ungraceful exits
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
    except RuntimeError:
        pass

    # Create a completely fresh execution context loop
    asyncio.set_event_loop(asyncio.new_event_loop())

    # Run the dashboard securely
    demo.launch(debug=True)
