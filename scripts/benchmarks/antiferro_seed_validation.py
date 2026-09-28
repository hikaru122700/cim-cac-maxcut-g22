"""日付付き反強磁性検証モジュールをプロジェクトルートから実行する。"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("antiferro_seed_study", ROOT / "modules/2026-09-28_AntiferroSeedStudy.py")
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)

if __name__ == "__main__":
    study.main()
