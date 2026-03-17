from transformers import pipeline
import gradio as gr

classifier = pipeline("sentiment-analysis")

def predict(text):
    result = classifier(text)[0]
    return f"Label: {result['label']}, Confidence: {result['score']:.2f}"

iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch()