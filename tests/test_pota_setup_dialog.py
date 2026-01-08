import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from PyQt5.QtWidgets import QApplication, QDialog

sys.path.append(str(Path(__file__).resolve().parents[1]))

from apps.PoTA.pota import PotaSetupDialog, build_pump_inputs_from_ui


def _ensure_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fill_valid_inputs(dialog: PotaSetupDialog):
    dialog.lineEdit_revolution.setText("1000")
    dialog.lineEdit_mass_flow.setText("1.5")
    dialog.lineEdit_outlet_pressure.setText("2.0")
    dialog.lineEdit_inlet_pressure.setText("1.0")
    dialog.lineEdit_inlet_angle.setText("10")
    dialog.lineEdit_inlet_diameter.setText("100")
    dialog.lineEdit_outlet_diameter.setText("200")
    dialog.lineEdit_density.setText("1000")
    dialog.lineEdit_dynamicViscosity.setText("1.0")
    dialog.lineEdit_vaporPressure.setText("100")


class TestPotaSetupDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _ensure_app()

    def setUp(self):
        self.dialog = PotaSetupDialog()

    def tearDown(self):
        self.dialog.close()

    def test_build_pump_inputs_converts_diameters_and_checkboxes(self):
        _fill_valid_inputs(self.dialog)
        self.dialog.checkBox_double_suction.setChecked(True)
        self.dialog.checkBox_second_stage.setChecked(False)

        rpm, pump_dict, fluid_props = build_pump_inputs_from_ui(self.dialog)

        self.assertEqual(rpm, 1000.0)
        self.assertAlmostEqual(pump_dict["inlet_radius"], 0.05)
        self.assertAlmostEqual(pump_dict["outlet_radius"], 0.1)
        self.assertTrue(pump_dict["double_suction"])
        self.assertFalse(pump_dict["second_stage"])
        self.assertEqual(fluid_props["density"], 1000.0)

    def test_accept_rejects_missing_required_fields(self):
        with mock.patch("apps.PoTA.pota.QMessageBox.warning"):
            self.dialog.accept()

        self.assertIsNone(self.dialog.pump)
        self.assertEqual(self.dialog.result(), QDialog.Rejected)

    def test_calc_pump_creates_pump(self):
        _fill_valid_inputs(self.dialog)
        self.dialog.checkBox_double_suction.setChecked(False)
        self.dialog.checkBox_second_stage.setChecked(False)

        result = self.dialog.calc_pump()

        self.assertTrue(result)
        self.assertIsNotNone(self.dialog.pump)


if __name__ == "__main__":
    unittest.main()
