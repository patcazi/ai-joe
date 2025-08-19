import gradio as gr
from ui.qa import answer


def chat_with_joe(message, history):
    hist_tuples = [(u or "", a or "") for (u, a) in (history or [])]
    return answer(message, hist_tuples)


def launch_ui():
    with gr.Blocks(title="AI Joe") as demo:
        gr.Markdown("# AI Joe")

        # Output chat — copy button enabled, larger height
        chatbot = gr.Chatbot(
            label="Conversation",
            height=600,
            show_copy_button=True,
        )

        # Bigger input box
        msg = gr.Textbox(
            placeholder="Ask me anything…",
            lines=6,
            autofocus=True,
            show_label=False,
        )

        # Store history in a State (list of (user, bot) tuples)
        history = gr.State([])

        send = gr.Button("Send", variant="primary")

        def on_submit(user_msg: str, hist):
            # Call existing backend, passing current history
            answer_text = chat_with_joe(user_msg, hist)
            # Append to history as (user, bot) tuple
            hist = (hist or []) + [(user_msg, answer_text)]
            return "", hist, hist

        # IMPORTANT: Chatbot is ONLY an output, never an input
        msg.submit(
            on_submit,
            inputs=[msg, history],
            outputs=[msg, history, chatbot],
            queue=True,
            api_name=False,
        )
        send.click(
            on_submit,
            inputs=[msg, history],
            outputs=[msg, history, chatbot],
            queue=True,
            api_name=False,
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)


if __name__ == "__main__":
    launch_ui()


