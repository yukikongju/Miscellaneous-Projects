from typing import List
from pydantic import BaseModel


class Document(BaseModel):
    title: str
    tags: List[str]
    content: str


def format_csv(doc: Document) -> str:
    return f"{doc.title}.csv"


def format_pdf(doc: Document) -> str:
    return f"{doc.title}.pdf"


def format_txt(doc: Document) -> str:
    return f"{doc.title}.txt"


def get_file_name(doc: Document, format_type: str) -> str:
    if format_type == "csv":
        file_name = format_csv(doc)
    elif format_type == "pdf":
        file_name = format_pdf(doc)
    elif format_type == "txt":
        file_name = format_txt(doc)
    else:
        raise NotImplementedError("{format_type} not implemented")

    return file_name


if __name__ == "__main__":
    doc = Document(
        title="Title",
        tags=["tag1", "tag2", "tag3"],
        content="Lorem ipsum dolor sit amet",
    )
    print(get_file_name(doc, "txt"))
    print(get_file_name(doc, "pdf"))
    print(get_file_name(doc, "csv"))
