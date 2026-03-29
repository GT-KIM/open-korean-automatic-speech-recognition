# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.
import re

NOISE_LABELS = ['/b', '/l', '/o', '/n', 'b/', 'l/', 'o/', 'n/', 'u/']
TIME_UNITS = ['년', '월', '일', '시', '분', '초', '대', '부대']

NUM_KOR1 = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
NUM_KOR2 = ["", "십", "백", "천"]
UNIT_BIG = ["", "만", "억", "조"]

def number_to_korean(num_str: str) -> str:
    num_str = num_str.lstrip('0')
    if not num_str:
        return "공"
    result = []
    num_len = len(num_str)
    split_unit = (num_len + 3) // 4
    num_str = num_str.zfill(split_unit * 4)

    for i in range(split_unit):
        part = num_str[i*4:(i+1)*4]
        part_result = []
        for j, digit in enumerate(part):
            d = int(digit)
            if d != 0:
                part_result.append(NUM_KOR1[d] + NUM_KOR2[3 - j])
        if part_result:
            result.append("".join(part_result) + UNIT_BIG[split_unit - i - 1])

    return "".join(result)

def normalize_text(text: str, remove_noise: bool = True, remove_unknown: bool = True) -> str:
    # 1. 이중전사: 영어면 앞, 아니면 발음(뒤)
    def replace_transcription(match):
        front, back = match.group(1), match.group(2)
        if re.fullmatch(r'[A-Za-z\s&\-.]+', front.strip()):
            return front.strip()
        else:
            return back.strip()

    text = re.sub(r'\(([^/]+)/([^)]+)\)', replace_transcription, text)

    # 2. 잡음 제거
    if remove_noise:
        for label in NOISE_LABELS:
            text = text.replace(label, '')

    # 3. 알아들을 수 없는 발화 제거
    if remove_unknown:
        text = re.sub(r'u/', '', text)

    # 4. 간투어 슬래시 제거
    text = re.sub(r'([가-힣])/', r'\1', text)

    # 5. 숫자 + 단위 분리: 1999년 → 1999 년
    for unit in TIME_UNITS:
        text = re.sub(rf'(\d+)\s*{unit}', rf'\1 {unit}', text)

    # 6. 모든 숫자 → 한글
    def replace_number(match):
        return number_to_korean(match.group())

    text = re.sub(r'\d+', replace_number, text)

    # 7. 문장 부호 정리 (허용된 것만)
    text = re.sub(r'[^\w\s,\.?!가-힣]', '', text)

    # 8. 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    text = "전화번호는 (867-860-2437)/(팔 육 칠 팔 육 공 에 이 사 삼 칠)이고, (UNESCO)/(유네코)는 중요한 기관이다. (5대)/(다섯 대)가 (24시간)/(스물 네 시간) 동안 일했다. (1999년)/(천 구백 구십 구 년)에 시작했다. /b 아/ 123123"
    print(normalize_text(text))