import os
import pathlib
import subprocess
import sys

from lift3d.helpers.common import Logger
import pathlib
import subprocess
import os
import signal
from contextlib import contextmanager

CAMERAS = [
    "corner",
    # "corner2",
]

TASKS = [
    # "assembly",
    # "button-press",
    # "box-close",
    # "drawer-open",
    # "reach",
    # "hammer",
    # "handle-pull",
    # "peg-unplug-side",
    "lever-pull",
    # "dial-turn",
    # "sweep-into",
    # "bin-picking",
    # "push-wall",
    # "hand-insert",
    # "shelf-place",
]

# 定义一个上下文管理器，用于启动和关闭 Xvfb
@contextmanager
def xvfb_display(display_num=99, screen_size="1024x768x24"):
    """启动 Xvfb 虚拟显示服务器，并在退出时关闭它"""
    xvfb_cmd = ["Xvfb", f":{display_num}", "-screen", "0", screen_size]
    xvfb_process = subprocess.Popen(xvfb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.environ["DISPLAY"] = f":{display_num}"
    try:
        yield
    finally:
        xvfb_process.terminate()
        xvfb_process.wait()

def main():
    code_root = pathlib.Path(__file__).resolve().parent.parent
    tool_path = code_root / "tools" / "gen_data_metaworld.py"
    
    # 使用 Xvfb 虚拟显示运行命令
    with xvfb_display(display_num=1, screen_size="1024x768x24"):
        for task in TASKS:
            for camera in CAMERAS:
                cmd = [
                    "python",  # 不再需要 `xvfb-run -a`，因为已经在 Xvfb 环境下
                    str(tool_path),
                    "--task-name", task,
                    "--camera-name", camera,
                    "--image-size", str(224),
                    "--num-episodes", str(100),
                    "--save-dir", "/workspaces/chenhao/data/metaworld",
                    "--episode-length", str(200),
                ]
                Logger.log_info(" ".join(cmd))
                subprocess.run(cmd)


if __name__ == "__main__":
    main()
