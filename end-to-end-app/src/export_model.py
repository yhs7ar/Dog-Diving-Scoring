"""Build the TensorRT ``.engine`` for the model of record (run this on the Jetson rig).

Thin CLI over :func:`dog_diving.detect.export_engine`, so the app's on-demand build and
this manual entry point run exactly the same code.  TensorRT engines are device-specific:
one built here will not run on another machine.
"""

from dog_diving.config import MODEL_PATH
from dog_diving.detect import export_engine


def export() -> None:
    print(f"Exporting {MODEL_PATH} to TensorRT engine...")
    engine = export_engine(MODEL_PATH, imgsz=640)
    print(f"Export complete: {engine}")


if __name__ == "__main__":
    export()
