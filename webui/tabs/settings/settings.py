import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from webui.tabs.settings.sections.presence import presence_tab
from webui.tabs.settings.sections.themes import theme_tab
from webui.tabs.settings.sections.lang import lang_tab
from webui.tabs.settings.sections.restart import restart_tab
from webui.tabs.settings.sections.model_author import model_author_tab
from webui.tabs.settings.sections.precision import precision_tab


def settings_tab():
    with gr.TabItem(label="General"):
        presence_tab()
        theme_tab()
        lang_tab()
        restart_tab()
    with gr.TabItem(label="Training"):
        model_author_tab()
        precision_tab()
