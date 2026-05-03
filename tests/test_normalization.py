import unittest

from openkoasr.normalization import normalize_text


class NormalizationTest(unittest.TestCase):
    def test_punctuation_agnostic(self):
        self.assertEqual(
            normalize_text(" Hello,   ASR! ", preset="punctuation_agnostic"),
            "hello asr",
        )

    def test_strict_keeps_punctuation(self):
        self.assertEqual(
            normalize_text(" 안녕,   하세요! ", preset="strict"),
            "안녕, 하세요!",
        )

    def test_kspon_numbers_and_noise(self):
        self.assertEqual(normalize_text("123", preset="kspon"), "일백이십삼")
        self.assertEqual(normalize_text("/b 안녕 u/", preset="kspon"), "안녕")


if __name__ == "__main__":
    unittest.main()
