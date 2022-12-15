import unittest
import sys

class GenericTests(unittest.TestCase):
    def test_module_import(self):
        import capsa
        from capsa import (
            ControllerWrapper,
            HistogramWrapper,
            VAEWrapper,
        )
        self.assertIn('capsa', sys.modules, "Capsa module not loaded")
        self.assertIn('capsa.controller_wrapper', sys.modules, "ControllerWrapper not loaded")
        self.assertIn('capsa.bias.histogram', sys.modules, "HistogramWrapper not loaded")
        self.assertIn('capsa.epistemic.vae', sys.modules, "VAEWrapper not loaded")
