"""The main Chat app."""

import reflex as rx
import reflex_chakra as rc

from chat.components import chat, navbar
from chat.state import  State

color = "rgb(107,99,246)"

def chat_page() -> rx.Component:
    """The main app."""
    return rc.vstack(
        navbar(),
        chat.chat(),
        chat.action_bar(),
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )

# Define the upload page
def index():
 return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select File",
                    color=color,
                    bg="white",
                    border=f"1px solid {color}",
                ),
                rx.text(
                    "Drag and drop files here or click to select files"
                ),
            ),
            id="upload1",
            border=f"1px dotted {color}",
            padding="5em",
        ),
        rx.hstack(
            rx.foreach(
                rx.selected_files("upload1"), rx.text
            )
        ),
        rx.button(
            "Upload",
            on_click=State.handle_upload(
                rx.upload_files(upload_id="upload1")
            ),
        ),

        rx.button(
            "Clear",
            on_click=rx.clear_selected_files("upload1"),
        ),
        rx.foreach(
            State.img,
            lambda img: rx.image(
                src=rx.get_upload_url(img)
            ),
        ),
        padding="5em",
    )

# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="green",
    ),
)
app.add_page(index)
app.add_page(chat_page, route="/chat")
