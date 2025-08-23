import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import named_entity_recognizer, CharacterNetworkGenerator
import os
import pandas as pd
from dotenv import load_dotenv
from character_chatbot import CharacterChatBot

load_dotenv()

# ---------------------------
# THEME CLASSIFICATION
# ---------------------------
def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove "dialogue" from the theme list if present
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    # Aggregate scores
    output_df = output_df.sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    # Return a chart
    return gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme", "Score"],
        vertical=False,
        width=500,
        height=260
    )

# ---------------------------
# CHARACTER NETWORK
# ---------------------------
def get_character_network(subtitles_path, ner_path):
    # Initialize with the required model path
    ner = named_entity_recognizer("en_core_web_md")  # change if you use another model
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html



def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatBot("khalil_akremi/breaking_bad_Llama-3-8B",
                                         huggingface_token = os.getenv('huggingface_token')
                                         )

    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output

# ---------------------------
# GRADIO APP
# ---------------------------
def main():
    with gr.Blocks() as iface:
        # -------------------
        # THEME CLASSIFICATION SECTION
        # -------------------
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path_theme = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(
                            get_themes,
                            inputs=[theme_list, subtitles_path_theme, save_path],
                            outputs=[plot]
                        )

        # -------------------
        # CHARACTER NETWORK SECTION
        # -------------------
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path_network = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="NERs Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(
                            get_character_network,
                            inputs=[subtitles_path_network, ner_path],
                            outputs=[network_html]
                        )
        
        # Character Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(chat_with_character_chatbot)
        # Launch the app
        iface.launch(share=True)


if __name__ == '__main__':
    main()
