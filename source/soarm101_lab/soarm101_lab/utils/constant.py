from pathlib import Path
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# soarm101_lab package root
PACKAGE_ROOT = Path(__file__).resolve().parents[4]

# assets root
ASSETS_ROOT = PACKAGE_ROOT / "assets"

ROBOT_URDF_PATH = str(ASSETS_ROOT / "SO101" / "urdf" / "so101_isaaclab.urdf")
TABLE_USD_PATH = str(ASSETS_ROOT / "table" / "usd" / "Danny_table.usd")
BIN_USD_PATH = str(ASSETS_ROOT / "bin" / "usd" / "bin_a02.usd")

# Danny Table
DANNY_TABLE_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Commercial/Tables/Danny.usd"

# Bin A02
BIN_A02_USD_PATH = "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/bin_a02/bin_a02.usd"