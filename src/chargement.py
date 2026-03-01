import frontmatter
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def preparer_chunks_depuis_markdown(dossier_data: str = "data") -> list[Document]:
    dossier = Path(dossier_data)
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks_finaux = []

    for chemin in dossier.glob("*.md"):
        post = frontmatter.load(chemin)
        segments = header_splitter.split_text(post.content)
        
        for i, seg in enumerate(segments):
            h1 = seg.metadata.get("Header 1", "")
            h2 = seg.metadata.get("Header 2", "")

            seg.page_content = f"{h1}\n{h2}\n{seg.page_content}"

            seg.metadata.update(post.metadata)
            seg.metadata["source"] = chemin.name
            seg.metadata["chunk_id"] = f"{chemin.name}::{h1}::{h2}::{i}"

            chunks_finaux.append(seg)
            
    return chunks_finaux