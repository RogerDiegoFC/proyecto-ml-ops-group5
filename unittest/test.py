import unittest
import requests
import os

API_URL = "http://localhost:8085/predict"
VALID_IMAGE = "number.png"   # imagen que sabes que funciona
INVALID_IMAGE = "invalid.txt"  # archivo inválido

class TestAPIModel(unittest.TestCase):

    def test_smoke(self):
        """Prueba de humo: Verificar que la API responde"""
        response = requests.get("http://localhost:8085/")
        self.assertEqual(response.status_code, 200)
        print("La API responde correctamente en /")

    def test_single_valid_prediction(self):
        """Prueba de un golpe: Enviar una sola imagen válida"""
        with open(VALID_IMAGE, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Posibe diagnostico 5", response.text)
        print("La predicción con una imagen válida fue correcta:", response.text)

    def test_edge_cases(self):
        """Prueba de borde: casos extremos"""
        response = requests.post(API_URL, files={"file": ("empty.png", b"")})
        self.assertNotEqual(response.status_code, 200)
        print("La API rechazó una imagen vacía")

        with open(INVALID_IMAGE, "w") as f:
            f.write("este no es un archivo de imagen")

        with open(INVALID_IMAGE, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        os.remove(INVALID_IMAGE)

        self.assertNotEqual(response.status_code, 200)
        print("La API rechazó un archivo inválido")

    def test_pattern_consistency(self):
        """Prueba de patrón: varias imágenes iguales deben dar mismo resultado"""
        results = []
        for _ in range(3):
            with open(VALID_IMAGE, "rb") as f:
                files = {"file": f}
                response = requests.post(API_URL, files=files)
                results.append(response.text)

        self.assertTrue(all(r == results[0] for r in results))
        print("La API fue consistente en respuestas repetidas:", results[0])



if __name__ == "__main__":
    unittest.main()

