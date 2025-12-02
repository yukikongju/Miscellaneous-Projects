"""
Implementing registry pattern using decorators
"""

from functools import wraps
from typing import List, Callable, Dict, Any
from pydantic import BaseModel


class Document(BaseModel):
    title: str
    tags: List[str]
    content: str


FormatFn = Callable[[Document], str]
formatters: Dict[str, FormatFn] = {}


def register_formatter(name: str):
    def decorator(func: FormatFn):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        formatters[name] = wrapper
        return wrapper

    return decorator


@register_formatter("csv")
def format_csv(doc: Document) -> str:
    return f"{doc.title}.csv"


@register_formatter("pdf")
def format_pdf(doc: Document) -> str:
    return f"{doc.title}.pdf"


@register_formatter("txt")
def format_txt(doc: Document) -> str:
    return f"{doc.title}.txt"


def get_file_name(doc: Document, format_type: str) -> str:
    formatter = formatters.get(format_type)
    if formatter is None:
        raise ValueError(f"Format {format_type} not implemented")
    return formatter(doc)


if __name__ == "__main__":
    doc = Document(
        title="Title",
        tags=["tag1", "tag2", "tag3"],
        content="Lorem ipsum dolor sit amet",
    )
    print(get_file_name(doc, "txt"))
    print(get_file_name(doc, "pdf"))
    print(get_file_name(doc, "csv"))
