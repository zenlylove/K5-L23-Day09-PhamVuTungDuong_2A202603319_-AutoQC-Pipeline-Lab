"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    remove_codes = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    by_image = defaultdict(list)
    for issue in issues:
        if issue.severity == ERROR:
            by_image[issue.image].append(issue)

    fixes = []
    removed = set()
    for stem, image_issues in by_image.items():
        if stem in removed:
            continue
        remove_issue = next((i for i in image_issues if i.code in remove_codes), None)
        if remove_issue is not None:
            image_path = ds.image_path(stem)
            label_path = ds.label_path(stem)
            if image_path is not None:
                image_path.unlink()
            label_path.unlink(missing_ok=True)
            fixes.append({
                "image": stem, "line": "", "action": "remove_image",
                "before": remove_issue.code, "after": "ảnh và nhãn đã loại khỏi dataset",
            })
            removed.add(stem)
            continue

        if stem == "data.yaml" and any(i.code == "BAD_CLASS_NAME" for i in image_issues):
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            old_names = data.get("names", {})
            data["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append({
                "image": "data.yaml", "line": "", "action": "rename_classes",
                "before": yaml.safe_dump(old_names, allow_unicode=True, sort_keys=False).strip(),
                "after": yaml.safe_dump(data["names"], allow_unicode=True, sort_keys=False).strip(),
            })
            continue

        if any(i.code == "MISSING_LABEL_FILE" for i in image_issues) and not ds.label_path(stem).exists():
            fixes.append({
                "image": stem, "line": "", "action": "needs_label",
                "before": "(không có file)", "after": "chuyển người gán",
            })

        label_path = ds.label_path(stem)
        if not label_path.exists():
            continue

        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        issues_by_line = defaultdict(list)
        for issue in image_issues:
            if issue.line is not None:
                issues_by_line[issue.line].append(issue)

        changed = False
        kept_lines = []
        for line_no, line in enumerate(lines, 1):
            line_issues = issues_by_line.get(line_no, [])
            drop_issue = next((i for i in line_issues if i.code in drop_codes), None)
            if drop_issue is not None:
                fixes.append({
                    "image": stem, "line": line_no, "action": "drop_line",
                    "before": line, "after": "",
                })
                changed = True
                continue

            clip_issue = next((i for i in line_issues if i.code == "BOX_OUT_OF_IMAGE"), None)
            if clip_issue is not None:
                try:
                    class_id, cx, cy, width, height = parse_yolo_line(line)
                except ValueError:
                    kept_lines.append(line)
                    continue

                x1, y1 = max(0.0, cx - width / 2), max(0.0, cy - height / 2)
                x2, y2 = min(1.0, cx + width / 2), min(1.0, cy + height / 2)
                if x2 <= x1 or y2 <= y1:
                    fixes.append({
                        "image": stem, "line": line_no, "action": "drop_line",
                        "before": line, "after": "",
                    })
                else:
                    clipped = (class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                    after = f"{clipped[0]} " + " ".join(f"{value:.8f}" for value in clipped[1:])
                    kept_lines.append(after)
                    fixes.append({
                        "image": stem, "line": line_no, "action": "clip_box",
                        "before": line, "after": after,
                    })
                changed = True
                continue

            kept_lines.append(line)

        if changed:
            label_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

    return fixes
