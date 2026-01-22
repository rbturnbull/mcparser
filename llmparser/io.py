from pathlib import Path
import yaml
from yaml.emitter import Emitter, ScalarAnalysis

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


def write_yaml(data, filepath:Path|str):
    with open(filepath, "w") as f:
        for file_key, data in data.items():
            yaml.dump({file_key: data}, f, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
            f.write("\n")                
