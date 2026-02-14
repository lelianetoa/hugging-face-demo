import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Chargement du modèle et du tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def predict(prompt):
    if not prompt or len(prompt.strip()) < 20:
        return "Texte trop court pour être résumé (minimum 20 caractères)."

    try:
        # 2. Préparation du texte (Tokenization)
        inputs = tokenizer(
            prompt, 
            max_length=1024, 
            return_tensors="pt", 
            truncation=True
        )

        # 3. Génération du résumé
        # On définit manuellement les paramètres de génération
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=30,
            max_length=150,
            early_stopping=True
        )

        # 4. Décodage du résultat
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

# 5. Interface Gradio
demo = gr.Interface(
    fn=predict, 
    inputs=gr.Textbox(lines=10, label="Collez votre texte ici", placeholder="Entrez le texte à résumer..."), 
    outputs=gr.Textbox(label="Résumé généré"),
    title="Assistant de Résumé Automatique",
    description="Cette application utilise DistilBART pour créer des résumés concis."
)

if __name__ == "__main__":
    demo.launch()