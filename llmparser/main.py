from pathlib import Path
import typer
import llmloader
import conllu
import yaml

app = typer.Typer()

@app.command()
def db(
    input:Path = typer.Argument(..., help="Path to input CoNLL-U file"),
    output:Path = typer.Argument(..., help="Path to output database"),
):
    # TODO Read CONLL-U file
    with open(input, "r", encoding="utf-8") as f:
        data = conllu.parse_incr(f)

        # TODO Create Document with metadata

        # TODO Embed sentences

    # TODO Save database

@app.command()
def parse(
    input:Path = typer.Argument(..., help="Path to input YAML file"),
    db:Path = typer.Option(..., help="Path to database with examples (see `db` command)"),
    output:Path = typer.Option(..., help="Path to output YAML file"),
    n_examples:int = typer.Option(100, help="Number of examples to include in the prompt"),
    model:str = typer.Option("claude-sonnet-4-5-20250929", help="LLM model to use"),
    temperature:float = typer.Option(0.2, help="LLM temperature"),
    api_key:str = typer.Option(None, help="API key for LLM"),
):
    llm = llmloader.load(model, temperature=temperature, api_key=api_key)

    # TODO Load database

    # TODO Read input YAML file
    data = 

    # Loop over sentences
    for key, item_data in data.items():
        # TODO Embed sentence

        # TODO Retrieve examples from database

        # TODO Create prompt
        prompt = ""
        chain = prompt | llm | StrOutputParser()

        # TODO Call LLM to get parse
        result = chain.invoke(prompt)

        # Store result
        item_data['conllu'] = result
        data[key] = item_data

    # Write output YAML file
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f)