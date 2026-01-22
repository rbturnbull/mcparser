
from .prompt import build_chain
from langchain_chroma import Chroma


def parse_items(
    data:dict,
    database:Chroma,
    llm,
    n_examples:int = 100,
) -> None:
    data = dict(data)  # Make a copy to avoid modifying the input dict
    
    # Loop over each item
    for key, item_data in data.items():
        assert 'text' in item_data
        sentence = item_data["text"]

        # Retrieve examples from database
        examples = database.similarity_search(sentence, k=n_examples)

        # Create prompt
        chain = build_chain(llm, sentence, examples)

        # Call LLM to get parsing
        result = chain.invoke({})

        # Store result
        item_data['conllu'] = result
        data[key] = item_data

    return data