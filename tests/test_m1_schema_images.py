import base64
import io

import pytest
from PIL import Image

from glance.images import load_image
from glance.schema import ImageLoadError, ImageRef, RequestValidationError, parse_request


def _paths(exc: RequestValidationError) -> list[str]:
    return [d["path"] for d in exc.detail]


def test_valid_request_parses(request_body):
    req = parse_request(request_body)
    assert req.model == "vlm"
    assert req.options.choice_method == "independent"
    assert req.options.calibrated is False
    assert list(req.questions) == ["is_red", "color", "brightness"]


@pytest.mark.parametrize(
    "mutate,expected_path",
    [
        (lambda b: b.update(model="gpt"), "model"),
        (lambda b: b["state"].update(images=[]), "state.images"),
        (lambda b: b["state"]["images"][0].update(path="x.jpg"), "state.images.0"),
        (lambda b: b["questions"]["color"].pop("criteria"), "questions.color.criteria"),
        (lambda b: b["questions"]["color"].update(criteria={"only": None}), "questions.color.criteria"),
        (lambda b: b["questions"]["brightness"].update(criteria=["one"]), "questions.brightness.criteria"),
        (lambda b: b["questions"]["brightness"].update(criteria=[str(i) for i in range(11)]), "questions.brightness.criteria"),
        (lambda b: b["questions"]["is_red"].update(instructions=""), "questions.is_red.instructions"),
        (lambda b: b["questions"]["is_red"].update(type="maybe"), "questions.is_red"),
        (lambda b: b.update(questions={}), "questions"),
        (lambda b: b.update(options={"choice_method": "magic"}), "options.choice_method"),
        (lambda b: b.update(extra_field=1), "extra_field"),
    ],
)
def test_invalid_bodies_name_the_field_path(request_body, mutate, expected_path):
    mutate(request_body)
    with pytest.raises(RequestValidationError) as err:
        parse_request(request_body)
    assert err.value.code == "validation_error" and err.value.http_status == 422
    assert expected_path in _paths(err.value)


def test_limits(request_body, png_b64):
    request_body["state"]["images"] = [{"id": f"img{i}", "base64": png_b64} for i in range(5)]
    with pytest.raises(RequestValidationError) as err:
        parse_request(request_body)
    assert "state.images" in _paths(err.value)

    request_body["state"]["images"] = [{"id": "img0", "base64": png_b64}]
    request_body["questions"]["color"]["criteria"] = {f"opt{i}": None for i in range(129)}
    with pytest.raises(RequestValidationError) as err:
        parse_request(request_body)
    assert "questions.color.criteria" in _paths(err.value)


def test_duplicate_image_ids_rejected(request_body, png_b64):
    request_body["state"]["images"].append({"id": "img0", "base64": png_b64})
    with pytest.raises(RequestValidationError):
        parse_request(request_body)


def test_non_dict_body_is_a_validation_error():
    with pytest.raises(RequestValidationError):
        parse_request(["not", "a", "request"])


# --- images -------------------------------------------------------------------------------------


def test_load_base64_and_data_url(cfg, png_b64):
    a = load_image(ImageRef(id="a", base64=png_b64), cfg.limits)
    b = load_image(ImageRef(id="b", base64="data:image/png;base64," + png_b64), cfg.limits)
    assert a.sha256 == b.sha256 and len(a.sha256) == 64
    assert (a.width, a.height, a.format, a.source) == (64, 48, "PNG", "base64")
    assert a.image.mode == "RGB"


def test_load_path_and_resize_policy(cfg, tmp_path):
    big = tmp_path / "big.jpg"
    Image.new("RGB", (4096, 1024), (10, 200, 10)).save(big)
    loaded = load_image(ImageRef(id="x", path=str(big)), cfg.limits)
    assert (loaded.width, loaded.height) == (4096, 1024)  # original size is what gets logged
    assert max(loaded.image.size) == cfg.limits.max_side


def test_bundled_sample_resolves_from_project_root(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_image(ImageRef(id="s", path="samples/receipt.jpg"), cfg.limits).format == "JPEG"


def test_rgba_is_flattened_on_white(cfg):
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buf, format="PNG")
    loaded = load_image(ImageRef(id="t", base64=base64.b64encode(buf.getvalue()).decode()), cfg.limits)
    assert loaded.image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(
    "ref",
    [
        ImageRef(id="m", path="/no/such/file.jpg"),
        ImageRef(id="b", base64="!!!not-base64!!!"),
        ImageRef(id="g", base64=base64.b64encode(b"these are not image bytes").decode()),
        ImageRef(id="u", url="ftp://example.com/x.jpg"),
    ],
)
def test_image_load_failures(cfg, ref):
    with pytest.raises(ImageLoadError) as err:
        load_image(ref, cfg.limits)
    assert err.value.code == "image_load_failed" and err.value.http_status == 400


def test_unsupported_format_and_size_limit(cfg, tmp_path):
    gif = tmp_path / "x.gif"
    Image.new("RGB", (8, 8)).save(gif)
    with pytest.raises(ImageLoadError, match="format"):
        load_image(ImageRef(id="g", path=str(gif)), cfg.limits)

    small_limit = cfg.limits.model_copy(update={"max_image_mb": 0})
    with pytest.raises(ImageLoadError, match="limit"):
        load_image(ImageRef(id="s", path="samples/receipt.jpg"), small_limit)
