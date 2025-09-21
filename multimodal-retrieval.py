import logging

import streamlit as st
import streamlit_cropper
from PIL import Image
from streamlit_cropper import st_cropper

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="wide")

from retrieval import MultimodalRetriever


def _recommended_box2(img: Image, aspect_ratio: tuple) -> dict:
    width, height = img.size
    return {
        "left": int(10),
        "top": int(10),
        "width": int(width - 30),
        "height": int(height - 30),
    }


@st.cache_resource
def init_retriever():
    retriever = MultimodalRetriever()
    return retriever


class MultimodalRetrieverUI:
    def __init__(self):
        streamlit_cropper._recommended_box = _recommended_box2
        self.retriever = init_retriever()
        self.init_sidebar()

    def init_sidebar(self):
        # logo
        st.sidebar.image("assets/sentinel_logo_white.png", use_column_width="always")

        # title
        st.title("SENTINEL Multimodal Retrieval")

        text_input = st.sidebar.text_input(
            "Query sentence", help="Insert the sentence/topic you want to to search for."
        )
        st.session_state["text_input"] = text_input
        if st.sidebar.button("Text-to-image search"):
            self.search("Text-to-image search")

        # this returns a file-like object
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpeg")
        if uploaded_file is not None:
            # Get a cropped image from the frontend
            uploaded_img = Image.open(uploaded_file)
            width, height = uploaded_img.size

            new_width = 370
            new_height = int((new_width / width) * height)
            uploaded_img = uploaded_img.resize((new_width, new_height))

            st.sidebar.text(
                "Query image",
                help="Edit the bounding box to change the ROI (Region of Interest).",
            )
            with st.sidebar.empty():
                cropped_img = st_cropper(
                    uploaded_img,
                    box_color="#4fc4f9",
                    realtime_update=True,
                    aspect_ratio=(16, 9),
                )
            st.session_state["cropped_img"] = cropped_img
            if st.sidebar.button("Image-to-image search"):
                self.search("Image-to-image search")

        show_captions = st.sidebar.toggle("Show captions", value=False)
        st.session_state["show_captions"] = show_captions

    def search(self, search_type: str):
        cols = st.columns(5)

        if search_type == "Text-to-image search":
            search_query = st.session_state.get("text_input")
        elif search_type == "Image-to-image search":
            search_query = st.session_state.get("cropped_img")
        else:
            raise ValueError(f"Invalid search type: {search_type}")

        # getting hits
        results = self.retriever.get_search_hits(search_query=search_query, top_k=30)

        for i, info in enumerate(results):
            imgName = info.filename
            caption = info.caption
            img = Image.open(imgName)
            cols[i % 5].image(img, use_column_width=True)
            if st.session_state.get("show_captions", None):
                cols[i % 5].write(f"{caption}")


if __name__ == "__main__":
    interface = MultimodalRetrieverUI()

# This Streamlit app is based on the great bootcamp tutorial from
# the Milvus team: https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/apps/multimodal_rag_with_milvus/ui.py
