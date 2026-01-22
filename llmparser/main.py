from pathlib import Path
import typer
import llmloader

from .io import write_yaml, load_file
from .database import load_database, build_database, DEFAULT_EMBEDDING_MODEL_ID
from .parsing import parse_items

app = typer.Typer()

@app.command()
def db(
    input:Path = typer.Argument(..., help="Path to input CoNLL-U file"),
    output:Path = typer.Argument(..., help="Path to output database"),
    embedding_model:str = typer.Option(DEFAULT_EMBEDDING_MODEL_ID, help="Embedding model ID"),
):
    build_database(
        conllu_path=input,
        output_path=output,
        embedding_model_id=embedding_model,
    )


@app.command()
def parse(
    input:Path = typer.Argument(..., help="Path to input YAML file"),
    db:Path = typer.Option(..., help="Path to database with examples (see `db` command)"),
    output:Path = typer.Option(..., help="Path to output YAML file"),
    n_examples:int = typer.Option(10, help="Number of examples to include in the prompt"),
    model:str = typer.Option("claude-sonnet-4-5-20250929", help="LLM model to use"),
    temperature:float = typer.Option(0.2, help="LLM temperature"),
    api_key:str = typer.Option(None, help="API key for LLM"),
    embedding_model:str = typer.Option(DEFAULT_EMBEDDING_MODEL_ID, help="Embedding model ID"),
):
    """ Parse items in input YAML file using LLM and example database, write output YAML file. """
    llm = llmloader.load(model, temperature=temperature, api_key=api_key)

    # Load database
    database = load_database(db, embedding_model_id=embedding_model)

    # Read input YAML file
    data = load_file(input)

    parsed_data = parse_items(data, database, llm, n_examples)

    # Write output YAML file
    write_yaml(parsed_data, output)


@app.command()
def convert_to_yaml(
    input:Path = typer.Argument(..., help="Path to input CoNLL-U file"),
    yaml_path:Path = typer.Argument(..., help="Path to output YAML file"),
) -> dict:
    from .io import convert_to_yaml as convert_to_yaml_func
    return convert_to_yaml_func(input, yaml_path)
