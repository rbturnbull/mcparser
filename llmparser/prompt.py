from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def clean_text(text:str) -> str:
    text = text.replace("```conllu", "").replace("```", "").strip()
    return text


def build_prompt(text:str, examples) -> ChatPromptTemplate:
    human_message = f"""
        You need to perform dependency parsing on this text:
        {text}

        These are examples of the CoNLL-U output that you need to produce:
    """
    for example in examples:
        human_message += "\n" + example["conllu"] + "\n"

    human_message += f"""
        Output exactly like those examples with the correct number of columns in the same format.
        When you finish the table, stop without outputting another character.
        Don't you dare output anything else.
        Don't provide metadata for the CoNLL-U, just the table itself.
        Start with the first line of the table. e.g. 1	
        Make sure you do not get any of the parsing wrong, otherwise you will be punished.

        Here is the text for you to parse with 100% accuracy:
        {text}
    """
    
    messages = [
        ("system", f"You are a computational linguist"),
        ("user", human_message),
    ]
    return ChatPromptTemplate.from_messages(messages=messages)


def build_chain(llm, text:str, examples):
    prompt = build_prompt(text, examples)
    chain = prompt | llm | StrOutputParser() | clean_text
    return chain
