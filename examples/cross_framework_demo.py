"""Example script demonstrating cross-framework experience transfer via UAS MemoryBridge.

Usage:
    python examples/cross_framework_demo.py
"""

from pathlib import Path
from uas.bridge.demo import run_bridge_interop_demo


def main() -> None:
    demo_output = Path("artifacts/bridge_demo")
    demo_output.mkdir(parents=True, exist_ok=True)
    result = run_bridge_interop_demo(demo_output, steps_to_train=100)
    print(f"\nDemo successfully finished! Summary: {result}")


if __name__ == "__main__":
    main()
