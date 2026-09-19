"""Renders the two synthetic sample documents. Everything here is made up and dedicated to the public domain (CC0).

Run: uv run python samples/make_samples.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def receipt() -> Image.Image:
    img = Image.new("RGB", (520, 900), (236, 232, 224))
    paper = Image.new("RGB", (400, 800), (252, 251, 247))
    d = ImageDraw.Draw(paper)
    y = 30
    d.text((200, y), "CORNER MARKET", font=font(30), fill=(20, 20, 20), anchor="ma")
    y += 44
    d.text((200, y), "214 Alder Street", font=font(18), fill=(60, 60, 60), anchor="ma")
    y += 26
    d.text((200, y), "Tel 555-0142", font=font(18), fill=(60, 60, 60), anchor="ma")
    y += 40
    d.line((24, y, 376, y), fill=(120, 120, 120), width=1)
    y += 16
    items = [("Whole milk 1L", "2.49"), ("Sourdough loaf", "5.25"), ("Bananas 0.8kg", "1.36"),
             ("Coffee beans 250g", "9.80"), ("Dish soap", "3.15"), ("Eggs x12", "4.60")]
    for name, price in items:
        d.text((28, y), name, font=font(20), fill=(25, 25, 25))
        d.text((372, y), price, font=font(20), fill=(25, 25, 25), anchor="ra")
        y += 34
    y += 8
    d.line((24, y, 376, y), fill=(120, 120, 120), width=1)
    y += 16
    for label, value, size in [("Subtotal", "26.65", 20), ("Tax 8%", "2.13", 20), ("TOTAL", "28.78", 28)]:
        d.text((28, y), label, font=font(size), fill=(15, 15, 15))
        d.text((372, y), value, font=font(size), fill=(15, 15, 15), anchor="ra")
        y += size + 16
    y += 12
    d.text((28, y), "VISA ****  APPROVED", font=font(18), fill=(60, 60, 60))
    y += 30
    d.text((28, y), "2026-03-14 18:42   Reg 03   #004417", font=font(16), fill=(60, 60, 60))
    y += 50
    d.text((200, y), "Thank you for shopping with us", font=font(18), fill=(40, 40, 40), anchor="ma")
    paper = paper.rotate(-2.5, expand=True, fillcolor=(236, 232, 224), resample=Image.BICUBIC)
    img.paste(paper, ((img.width - paper.width) // 2, (img.height - paper.height) // 2))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def invoice() -> Image.Image:
    img = Image.new("RGB", (850, 1100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 850, 110), fill=(32, 58, 96))
    d.text((50, 34), "INVOICE", font=font(44), fill=(255, 255, 255))
    d.text((800, 30), "Northfield Design Studio", font=font(22), fill=(255, 255, 255), anchor="ra")
    d.text((800, 62), "88 Harbor Road, Suite 4", font=font(18), fill=(215, 225, 240), anchor="ra")
    d.text((50, 150), "Bill to:", font=font(18), fill=(110, 110, 110))
    d.text((50, 176), "Lakeside Bakery LLC", font=font(24), fill=(20, 20, 20))
    d.text((50, 208), "Attn: Accounts Payable", font=font(18), fill=(60, 60, 60))
    d.text((800, 150), "Invoice no. INV-2031", font=font(20), fill=(20, 20, 20), anchor="ra")
    d.text((800, 180), "Issued 2026-02-01", font=font(18), fill=(60, 60, 60), anchor="ra")
    d.text((800, 206), "Payment due 2026-03-03", font=font(18), fill=(160, 40, 40), anchor="ra")
    y = 290
    d.rectangle((50, y, 800, y + 40), fill=(235, 239, 245))
    for x, label, anchor in [(62, "Description", "la"), (560, "Hours", "ra"), (670, "Rate", "ra"), (788, "Amount", "ra")]:
        d.text((x, y + 9), label, font=font(18), fill=(40, 40, 40), anchor=anchor)
    y += 56
    rows = [("Brand identity refresh", "24", "95.00", "2,280.00"), ("Menu layout and print files", "10", "95.00", "950.00"),
            ("Storefront signage mockups", "6", "95.00", "570.00")]
    for desc, hours, rate, amount in rows:
        d.text((62, y), desc, font=font(20), fill=(25, 25, 25))
        d.text((560, y), hours, font=font(20), fill=(25, 25, 25), anchor="ra")
        d.text((670, y), rate, font=font(20), fill=(25, 25, 25), anchor="ra")
        d.text((788, y), amount, font=font(20), fill=(25, 25, 25), anchor="ra")
        y += 44
        d.line((50, y - 10, 800, y - 10), fill=(225, 225, 225), width=1)
    y += 20
    d.text((670, y), "Amount due", font=font(24), fill=(20, 20, 20), anchor="ra")
    d.text((788, y), "$3,800.00", font=font(24), fill=(20, 20, 20), anchor="ra")
    y += 90
    d.text((50, y), "Please remit payment within 30 days to:", font=font(18), fill=(60, 60, 60))
    d.text((50, y + 28), "Northfield Design Studio, account ending 0000 (sample data)", font=font(18), fill=(60, 60, 60))
    return img


if __name__ == "__main__":
    receipt().save(OUT / "receipt.jpg", quality=90)
    invoice().save(OUT / "invoice.jpg", quality=90)
    print("wrote receipt.jpg, invoice.jpg")
