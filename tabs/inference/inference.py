import os, sys
import gradio as gr
import regex as re
import shutil
import datetime
import random

from core import (
    run_infer_script,
    run_batch_infer_script,
)

from assets.i18n.i18n import I18nAuto

from rvc.lib.utils import format_title

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")

model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and root == audio_root_relative
    and "_output" not in name
]


def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(os.path.dirname(input_audio_path), new_name)
    return output_path


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, output_path_fn(target_path)


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)


def delete_outputs():
    gr.Info(f"Outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


def match_index(model_file_value):
    if model_file_value:
        model_folder = os.path.dirname(model_file_value)
        index_files = get_indexes()
        for index_file in index_files:
            if os.path.dirname(index_file) == model_folder:
                return index_file
    return ""


# Inference tab
def inference_tab():
    default_weight = random.choice(names) if names else None
    with gr.Row():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(names, key=lambda path: os.path.getsize(path)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )

            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the .INDEX file to use for the conversion."),
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Column():
            refresh_button = gr.Button(i18n("Refresh"))
            unload_button = gr.Button(i18n("Unload Voice"))

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    # Single inference tab
    with gr.Tab(i18n("Single")):
        with gr.Column():
            upload_audio = gr.Audio(
                label=i18n("Upload Audio"), type="filepath", editable=False
            )
            with gr.Row():
                audio = gr.Dropdown(
                    label=i18n("Select Audio"),
                    info=i18n("Select the audio to convert."),
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Accordion(i18n("Settings"), open=False):
            with gr.Column():
                clear_outputs_infer = gr.Button(
                    i18n("Clear Outputs (Deletes all audios in assets/audios)")
                )
                output_path = gr.Textbox(
                    label=i18n("Output Path"),
                    placeholder=i18n("Enter output path"),
                    info=i18n(
                        "The path where the output audio will be saved. By default in assets/audios/output.wav"
                    ),
                    value=(
                        output_path_fn(audio_paths[0])
                        if audio_paths
                        else os.path.join(now_dir, "assets", "audios", "output.wav")
                    ),
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label=i18n("Split Audio"),
                    info=i18n(
                        "Splits audio in chunks. Brings quicker results & fixes inconsistent volume bug."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    label=i18n("Autotune"),
                    info=i18n(
                        "Applies soft autotune. Ideal for singing audios."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_audio = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    info=i18n(
                        "De-noises the output. Ideal for speaking audios."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Clean Strength"),
                    info=i18n(
                        "Define clean-up level. Very high values may compress the output too much."
                    ),
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    info=i18n(
                        "Set the pitch for the voice. Higher values increase the pitch."
                    ),
                    value=0,
                    interactive=True,
                )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n("Filter Radius"),
                    info=i18n(
                        "A value of 3 or higher applies median filtering. May reduce respiration. Works if you use Harvest."
                    ),
                    value=3,
                    step=1,
                    interactive=True,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Index Rate"),
                    info=i18n(
                        "Level of influence of model's .INDEX. Lowering it may minimize artifacts."
                    ),
                    value=0.75,
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Volume Envelope"),
                    info=i18n(
                        "The closer to 0, the more the audio will keep its original volume. The closer to 1, it will match the loudness of the model."
                    ),
                    value=1,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protection"),
                    info=i18n(
                        "Suppresses breath sounds to reduce artifacting. Lowering it too much may suppress some speech."
                    ),
                    value=0.5,
                    interactive=True,
                )
                hop_length = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label=i18n("Hop Length"),
                    info=i18n(
                        "Defines the duration it takes the voice hit a significant pitch change. Smaller values take longer to process, but yield a more precise pitch."
                    ),
                    value=128,
                    interactive=True,
                )
            with gr.Column():
                f0method = gr.Radio(
                    label=i18n("Pitch Extraction Algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. RMVPE is the most recommended."
                    ),
                    choices=[
                        "pm",
                        "harvest",
                        "dio",
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                        "hybrid[rmvpe+fcpe]",
                    ],
                    value="rmvpe",
                    interactive=True,
                )

        convert_button1 = gr.Button(i18n("Convert"))

        with gr.Row():  # Defines output info + output audio download after conversion
            vc_output1 = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )
            vc_output2 = gr.Audio(label=i18n("Export Audio"))

    # Batch inference tab
    with gr.Tab(i18n("Batch")):
        with gr.Row():
            with gr.Column():
                input_folder_batch = gr.Textbox(
                    label=i18n("Input Folder"),
                    info=i18n("Select the folder containing the audios."),
                    placeholder=i18n("Enter input path"),
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
                output_folder_batch = gr.Textbox(
                    label=i18n("Output Folder"),
                    info=i18n(
                        "Select the folder for the results."
                    ),
                    placeholder=i18n("Enter output path"),
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
        with gr.Accordion(i18n("Settings"), open=False):
            with gr.Column():
                clear_outputs_batch = gr.Button(
                    i18n("Clear Outputs (Deletes all audios in assets/audios)")
                )
                split_audio_batch = gr.Checkbox(
                    label=i18n("Split Audio"),
                    info=i18n(
                        "Splits audio in chunks. Brings quicker results & fixes inconsistent volume bug."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_batch = gr.Checkbox(
                    label=i18n("Autotune"),
                    info=i18n(
                        "Applies soft autotune. Ideal for singing audios."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_audio_batch = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    info=i18n(
                        "De-noises the output. Ideal for speaking audios."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Clean Strength"),
                    info=i18n(
                        "Define clean-up level. Very high values may compress the output too much."
                    ),
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    info=i18n(
                        "Set the pitch for the voice. Higher values increase the pitch."
                    ),
                    value=0,
                    interactive=True,
                )
                filter_radius_batch = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n("Filter Radius"),
                    info=i18n(
                        "Level of influence of model's .INDEX. Lowering it may minimize artifacts"
                    ),
                    value=3,
                    step=1,
                    interactive=True,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Index Rate"),
                    info=i18n(
                        "Level of influence of model's .INDEX. Lowering it may minimize artifacts."
                    ),
                    value=0.75,
                    interactive=True,
                )
                rms_mix_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Volume Envelope"),
                    info=i18n(
                        "The closer to 0, the more the audio will keep its original volume. The closer to 1, it will match the loudness of the model."
                    ),
                    value=1,
                    interactive=True,
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protection"),
                    info=i18n(
                        "Suppresses breath sounds to reduce artifacting. Lowering it too much may suppress some speech."
                    ),
                    value=0.5,
                    interactive=True,
                )
                hop_length_batch = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label=i18n("Hop Length"),
                    info=i18n(
                        "Defines the duration it takes the voice hit a significant pitch change. Smaller values take longer to process, but yield a more precise pitch."
                    ),
                    value=128,
                    interactive=True,
                )
            with gr.Column():
                f0method_batch = gr.Radio(
                    label=i18n("Pitch Extraction Algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. RMVPE is the most recommended."
                    ),
                    choices=[
                        "pm",
                        "harvest",
                        "dio",
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                        "hybrid[rmvpe+fcpe]",
                    ],
                    value="rmvpe",
                    interactive=True,
                )

        convert_button2 = gr.Button(i18n("Convert"))

        with gr.Row():  # Defines output info + output audio download after conversion
            vc_output3 = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    clean_audio_batch.change(
        fn=toggle_visible,
        inputs=[clean_audio_batch],
        outputs=[clean_strength_batch],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[],
        outputs=[model_file, index_file, audio],
    )
    audio.change(
        fn=output_path_fn,
        inputs=[audio],
        outputs=[output_path],
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    convert_button1.click(
        fn=run_infer_script,
        inputs=[
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0method,
            audio,
            output_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            clean_audio,
            clean_strength,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button2.click(
        fn=run_batch_infer_script,
        inputs=[
            pitch_batch,
            filter_radius_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            hop_length_batch,
            f0method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            autotune_batch,
            clean_audio_batch,
            clean_strength_batch,
        ],
        outputs=[vc_output3],
    )
