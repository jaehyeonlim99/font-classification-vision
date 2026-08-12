from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FONT_DIR = PROJECT_ROOT / "dataset" / "font"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "synthetic_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


fonts = {
    "AJumMa": FONT_DIR / "AJumMa.ttf",
    "GangIn": FONT_DIR / "GangIn.ttf",
    "GgeuTeuMeoRi": FONT_DIR / "GgeuTeuMeoRi.ttf",
    "GoRyeo": FONT_DIR / "GoRyeo.ttf",
    "SanHae": FONT_DIR / "SanHae.ttf",
    "SoMi": FONT_DIR / "SoMi.ttf",
    "Wild": FONT_DIR / "Wild.ttf",
    "YeorSa": FONT_DIR / "YeorSa.ttf"
}


texts = [
    "오늘도 한 걸음 천천히 앞으로 간다",
    "조용한 오후의 풍경이 마음을 편안하게 한다",
    "컴퓨터 비전 프로젝트를 다시 설계한다",
    "작은 노력의 반복이 큰 결과를 만든다",
    "노트북 화면의 글꼴을 분류하는 실험을 한다",
    "가을 낙엽이 떨어지는 길을 천천히 걷는다",
    "햇살이 비치는 창가에서 책장을 넘긴다",
    "배움은 끝이 없고 오늘도 새로 시작한다",
    "조용한 음악과 함께 생각을 정리해 본다",
    "폰트의 획과 곡선 모양을 분석해 본다",
    "이미지를 여러 조각으로 나누어 예측한다",
    "모델 성능은 데이터 품질에 크게 좌우된다",
    "실제 촬영 환경에 맞춰 데이터를 다시 만든다",
    "작은 차이가 분류 성능에 큰 영향을 준다",
    "천천히 하지만 확실하게 프로젝트를 완성한다"
]


def make_text_block():

    line_count = random.choice([1, 2])
    selected = random.sample(texts, line_count)

    return selected


def random_background_color():

    candidates = [
        (255,255,255),
        (250,248,240),
        (245,245,238),
        (248,246,230),
        (242,242,242)
    ]

    return random.choice(candidates)


def main():

    images_per_font = 800

    CANVAS_W = 1600
    CANVAS_H = 900

    for font_name, font_path in fonts.items():

        save_dir = OUTPUT_DIR / font_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"{font_name} 시작")

        for i in range(images_per_font):

            bg = random_background_color()

            img = Image.new("RGB",(CANVAS_W,CANVAS_H),bg)
            draw = ImageDraw.Draw(img)

            font_size = random.randint(140,260)
            line_spacing = random.randint(40,80)

            font = ImageFont.truetype(str(font_path),font_size)

            lines = make_text_block()

            x = random.randint(200,600)
            y = random.randint(200,500)

            current_y = y

            for line in lines:

                draw.text((x,current_y),line,font=font,fill="black")

                current_y += font_size + line_spacing

            angle = random.uniform(-2,2)

            img = img.rotate(angle,fillcolor=bg)

            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(random.uniform(0,1)))

            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9,1.1))

            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.9,1.1))

            img.save(save_dir / f"{i}.png")

        print(f"{font_name} 완료")


if __name__ == "__main__":
    main()