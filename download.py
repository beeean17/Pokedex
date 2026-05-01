import kagglehub
import shutil
from pathlib import Path


path = kagglehub.dataset_download("lantian773030/pokemonclassification")
source_path = Path(path)
dataset_path = source_path / "PokemonData"
if not dataset_path.exists():
    dataset_path = source_path

target_path = Path(__file__).resolve().parent / "PokemonData"

if target_path.exists():
    base_path = target_path
    index = 1
    while target_path.exists():
        target_path = base_path.with_name(f"{base_path.name}_{index}")
        index += 1

shutil.move(str(dataset_path), str(target_path))

print("Dataset moved to:", target_path)
