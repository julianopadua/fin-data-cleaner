import os
import yaml

def load_config():
    """Carrega config.yaml do diretório raiz e resolve caminhos relativos."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    def _p(rel):
        # remove prefixo ./ se existir e junta com a raiz do projeto
        rel = rel.lstrip("./")
        return os.path.join(project_root, rel)

    paths = {
        "project_root": project_root,
        "data_raw": _p(config["paths"]["data_raw"]),
        "data_processed": _p(config["paths"]["data_processed"]),
        "images": _p(config["paths"]["images"]),
        "report": _p(config["paths"]["report"]),
        "addons": _p(config["paths"]["addons"]),
    }

    return paths, config
