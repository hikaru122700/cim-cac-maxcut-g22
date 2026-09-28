"""反強磁性検証の保存済み結果へ、上書きせず独立乱数追試を追加する。"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("antiferro_seed_diagnostics", ROOT/"modules/2026-09-28_AntiferroSeedDiagnostics.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        raise SystemExit("保存先ディレクトリを1個指定してください")
    module.diagnose(Path(sys.argv[1]))
