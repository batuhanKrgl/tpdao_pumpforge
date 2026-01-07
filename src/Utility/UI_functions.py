import json

from PyQt5 import Qt
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QFileDialog


def is_line_edits_empty(self):
    line_edits = self.findChildren(QLineEdit)

    # Boş satır sayacını tut
    empty_count = 0

    # QLineEdit satırlarındaki değerleri kontrol et
    values = []
    for lineEdit in line_edits:
        value = lineEdit.text().strip()
        if not value:
            empty_count += 1
            lineEdit.setStyleSheet("background-color: pink;")
        else:
            values.append(float(value))
            lineEdit.setStyleSheet("")  # Arka plan rengini sıfırla

    if empty_count > 0:
        QMessageBox.warning(self, "Girdi Alanları Boş Bırakılamaz",
                            "Kırmızı ile işaretli alanlar boş bırakılamaz. Lütfen doldurun.")
        return True
    else:
        return False


def import_json_file_as_dict(self=None, caption=None, init_dir=None, file_type=None):
    if self is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        file_path = askopenfilename()
    else:
        file_path, _ = QFileDialog.getOpenFileName(parent=self, caption=caption, directory=init_dir, filter=file_type)
    try:
        with open(file_path, "r") as file:
            return json.load(file), True

    except:
        QMessageBox.warning(self, "Dosya Okunurken Bir Hata Oluştu.",
                            "Seçilen dosyanın '.json' formatında olduğundan emin olunuz.")
        return {}, False
