import os
import sys


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    try:
        return os.path.join(base_path[:base_path.index("src")], relative_path)
    except:
        return os.path.join(base_path, relative_path)
