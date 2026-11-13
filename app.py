"""Web interface (Gradio). Run:  python app.py   then open http://127.0.0.1:7860"""
import gradio as gr

from vocab import Vocab
 
v = Vocab()


def get_definition(word):
    word = word.strip()
    if not word:
        return "", "Type a word first."
    if word in v.data:
        return v.data[word]["definition"], "Already in your list. Definition loaded."
    return v.define(word), "Check the definition and edit it if it is wrong."


def make_sentence(word, definition):
    word, definition = word.strip(), definition.strip()
    if not word or not definition:
        return "", gr.update(choices=[], value=[]), "Need a word and a definition."
    v.add_word(word, definition)             # saves the (possibly edited) definition
    review = v.pick_review(word)
    sentence = v.make_sentence(word, review)
    return sentence, gr.update(choices=[word] + review, value=[]), "Copy the sentence to your notebook."


def save_unknown(unknown):
    v.mark_unknown(unknown)
    # also clear all fields, so the next word can be typed right away
    return "", "", "", gr.update(choices=[], value=[]), \
        f"Saved. Marked as unknown: {', '.join(unknown) or 'nothing'}."


with gr.Blocks(title="Vocabulary sentence trainer") as demo:
    gr.Markdown("# Vocabulary sentence trainer\n"
                "Add a word, get one natural sentence that also reviews words you may forget.")
    word = gr.Textbox(label="New word or phrase")
    btn_def = gr.Button("1. Get definition")
    definition = gr.Textbox(label="Definition (edit if wrong)", lines=2)
    btn_sent = gr.Button("2. Make sentence")
    sentence = gr.Textbox(label="Sentence", lines=3)
    unknown = gr.CheckboxGroup(label="Which of these words did you NOT know?")
    btn_save = gr.Button("3. Save")
    status = gr.Markdown()

    btn_def.click(get_definition, word, [definition, status])
    btn_sent.click(make_sentence, [word, definition], [sentence, unknown, status])
    btn_save.click(save_unknown, unknown, [word, definition, sentence, unknown, status])

if __name__ == "__main__":
    demo.launch()
