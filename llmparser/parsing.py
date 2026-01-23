from pathlib import Path
from rich.progress import track
from langchain_chroma import Chroma

from .io import write_yaml
from .prompt import build_chain

def parse_items(
    data:dict,
    database:Chroma,
    llm,
    n_examples:int = 10,
    output:str|Path|None = None,
) -> dict:
    data = dict(data)  # Make a copy to avoid modifying the input dict
    
    # Loop over each item
    for key, item_data in track(data.items(), description="Parsing items", total=len(data)):
        assert 'text' in item_data
        sentence = item_data["text"]

        # Retrieve examples from database
        examples = database(sentence, k=n_examples)

        # Create prompt
        chain = build_chain(llm, sentence, examples)

        # Call LLM to get parsing
        result = chain.invoke({})

        # Store result
        item_data['conllu'] = result
        data[key] = item_data

        # Write output YAML file
        if output is not None:
            write_yaml(data, output)

    return data