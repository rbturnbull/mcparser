from pathlib import Path
import yaml
from yaml.emitter import Emitter, ScalarAnalysis
import conllu
from rich.progress import track


# Keep original function
_orig_analyze_scalar = Emitter.analyze_scalar

def analyze_scalar_allow_block(self, scalar):
    analysis = _orig_analyze_scalar(self, scalar)
    # For multi-line scalars, force block style to be allowed
    if isinstance(analysis, ScalarAnalysis) and '\n' in scalar:
        analysis.allow_block = True
    return analysis

Emitter.analyze_scalar = analyze_scalar_allow_block

class LiteralDumper(yaml.SafeDumper):
    pass

def str_presenter(dumper, data):
    if "\n" in data:
        # Force literal block scalar (|)
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

LiteralDumper.add_representer(str, str_presenter)


def yaml_to_dict(filepath:Path|str):
    return yaml.parse( Path(filepath).read_text() )    


def write_yaml(data, filepath:Path|str):
    with open(filepath, "w") as f:
        for file_key, data in data.items():
            yaml.dump({file_key: data}, f, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
            f.write("\n")                


def conllu_to_dict(
    conllu_path:Path,
) -> dict:
    output_data = {}
    
    with open(conllu_path, "r", encoding="utf-8") as f:
        data = conllu.parse_incr(f)

        for example in track(data):
            item_data = dict(example.metadata)
            sent_id = next(item_data.values())
            text = item_data.get("text", "").strip()
            if not text:
                continue            
            item_data["text"] = text
            output_data[sent_id] = item_data

    return output_data


def convert_to_yaml(input:Path, yaml_path:Path) -> dict:
    output_data = load_file(input)    
    write_yaml(output_data, yaml_path)
    
    return output_data


def load_file(filepath:Path|str):
    suffix = Path(filepath).suffix.lower()
    if suffix in [".yaml", ".yml"]:
        return yaml_to_dict(filepath)
    elif suffix == ".conllu":
        return conllu_to_dict(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")