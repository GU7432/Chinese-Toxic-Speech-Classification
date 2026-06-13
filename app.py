"""中文毒性言論偵測 — 互動式展示 (Gradio)。

載入微調後的 ToxiCN 二分類模型，提供一個精緻的網頁介面：

* 模型載入：在介面中自行指定本機檢查點資料夾並按下「載入模型」（**啟動時不會自動載入**，
  方便剛 git clone、尚無模型檔的使用者選擇自己的模型）。
* 單句偵測：輸入中文文字 → 即時給出「有毒 / 非毒」判定、信心度長條與可調閾值。
* 批次偵測：貼上多行文字 → 一次輸出表格與整體統計。

於本機 GPU 啟動：

    uv run python app.py

可用 ``--checkpoint`` 或環境變數 ``TOXICN_CKPT`` 預先填入路徑（僅填入輸入框，仍需手動按
「載入模型」）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

LABELS_ZH = ["非毒", "有毒"]

# 展示用範例：涵蓋友善、不同毒性類型與隱晦表達。
EXAMPLES = [
    ["今天天氣真好，我們一起去公園散步吧！"],
    ["謝謝你的幫忙，這份報告寫得很用心。"],
    ["你這種人就是社會的垃圾，廢物一個。"],
    ["這個地方的人素質都很差，難怪這麼窮。"],
    ["呵呵，就你這智商也配跟我談？"],
    ["我覺得這部電影的節奏有點慢，但攝影很棒。"],
]


# ---------------------------------------------------------------------------
# 模型載入與推論
# ---------------------------------------------------------------------------


class ToxicClassifier:
    """封裝分詞器與模型，提供單句機率輸出。"""

    def __init__(self, checkpoint: Path, max_length: int = 512, no_cuda: bool = False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.checkpoint = checkpoint
        self.max_length = max_length
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and not no_cuda) else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def proba(self, text: str) -> float:
        """回傳 p(有毒)。"""
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1].cpu())


# 全域分類器；啟動時為 None，必須由使用者於介面中載入。
CLF: ToxicClassifier | None = None


def read_macro_f1(checkpoint: Path) -> float | None:
    """嘗試從檢查點同層的 metrics.json 讀取 test macro-F1。"""
    metrics = checkpoint.parent / "metrics.json"
    if not metrics.exists():
        return None
    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
        return float(data["test_metrics"]["macro_f1"])
    except Exception:
        return None


def load_model(checkpoint_path: str, max_length: int, no_cuda: bool) -> str:
    """載入使用者指定的檢查點；回傳狀態 Markdown。失敗時將 CLF 設回 None。"""
    global CLF
    raw = (checkpoint_path or "").strip().strip('"').strip("'")
    if not raw:
        CLF = None
        return "⚠️ 請先輸入檢查點資料夾路徑，再按「載入模型」。"

    p = Path(raw).expanduser()
    if p.is_file():  # 容許指向資料夾內任一檔案（例如 config.json）
        p = p.parent
    if not p.exists():
        CLF = None
        return f"❌ 路徑不存在：`{p}`"
    if not (p / "config.json").exists():
        CLF = None
        return f"❌ `{p}` 不是有效的檢查點資料夾（缺少 `config.json`）。"

    try:
        CLF = ToxicClassifier(p, max_length=max_length, no_cuda=no_cuda)
    except Exception as exc:  # noqa: BLE001 - 將任何載入錯誤回報到 UI
        CLF = None
        return f"❌ 載入失敗：{exc}"

    mf1 = read_macro_f1(p)
    extra = f"　·　test macro-F1 **{mf1:.3f}**" if mf1 is not None else ""
    return f"✅ 已載入模型：`{p}`　·　裝置 `{CLF.device}`{extra}"


def on_browse(selected) -> str:
    """檔案瀏覽器選取後，將路徑（檔案則取其資料夾）填入輸入框。"""
    if not selected:
        return gr.update()
    sel = selected[0] if isinstance(selected, (list, tuple)) else selected
    p = Path(sel)
    if p.is_file():
        p = p.parent
    return str(p)


# ---------------------------------------------------------------------------
# UI 回呼
# ---------------------------------------------------------------------------

_VERDICT_TEMPLATE = """
<div style="border-radius:16px;padding:20px 24px;margin-top:4px;
            background:{bg};border:1px solid {border};">
  <div style="font-size:30px;font-weight:800;color:{fg};line-height:1.2;">
    {emoji} {label}
  </div>
  <div style="font-size:15px;color:{fg};opacity:.85;margin-top:6px;">
    p(有毒) = {prob:.3f}　·　判定閾值 {threshold:.2f}
  </div>
</div>
"""

_NEEDS_MODEL_HTML = (
    "<div style='border-radius:16px;padding:20px 24px;margin-top:4px;"
    "background:#fffbe6;border:1px solid #ffe58f;color:#ad6800;font-size:16px;'>"
    "ℹ️ 尚未載入模型。請先在上方「模型」區指定檢查點並按「載入模型」。</div>"
)


def _verdict_html(prob: float, threshold: float) -> str:
    toxic = prob >= threshold
    if toxic:
        style = dict(bg="#fff1f0", border="#ffccc7", fg="#a8071a", emoji="☠", label="有毒")
    else:
        style = dict(bg="#f6ffed", border="#b7eb8f", fg="#237804", emoji="✅", label="非毒")
    return _VERDICT_TEMPLATE.format(prob=prob, threshold=threshold, **style)


def classify_single(text: str, threshold: float):
    """單句偵測 → (判定 HTML, 信心度 Label)。"""
    if CLF is None:
        return _NEEDS_MODEL_HTML, {}
    text = (text or "").strip()
    if not text:
        empty = "<div style='padding:16px;color:#888;'>請先輸入一段文字。</div>"
        return empty, {}
    prob = CLF.proba(text)
    confidences = {f"{LABELS_ZH[1]} (toxic)": prob, f"{LABELS_ZH[0]} (non-toxic)": 1.0 - prob}
    return _verdict_html(prob, threshold), confidences


def classify_batch(blob: str, threshold: float):
    """批次偵測 → (結果表格, 統計摘要 Markdown)。"""
    if CLF is None:
        return [], "ℹ️ 尚未載入模型。請先在上方「模型」區載入檢查點。"
    lines = [ln.strip() for ln in (blob or "").splitlines() if ln.strip()]
    if not lines:
        return [], "請貼上至少一行文字。"
    rows = []
    toxic_n = 0
    for ln in lines:
        p = CLF.proba(ln)
        is_tox = p >= threshold
        toxic_n += int(is_tox)
        rows.append([ln, "☠ 有毒" if is_tox else "✅ 非毒", round(p, 3)])
    total = len(rows)
    summary = (
        f"**共 {total} 句**　·　有毒 **{toxic_n}**（{toxic_n / total:.0%}）"
        f"　·　非毒 **{total - toxic_n}**（{(total - toxic_n) / total:.0%}）"
        f"　·　閾值 {threshold:.2f}"
    )
    return rows, summary


# ---------------------------------------------------------------------------
# 介面組裝
# ---------------------------------------------------------------------------


def build_demo(default_ckpt: str, max_length: int, no_cuda: bool,
               browse_root: str) -> gr.Blocks:
    with gr.Blocks(title="中文毒性言論偵測") as demo:
        gr.Markdown(
            "# ☠️ 中文毒性言論偵測\n"
            "微調 **hfl/chinese-roberta-wwm-ext** 於 **ToxiCN**（ACL 2023）資料集的二分類模型"
        )

        # 跨分頁共用的執行階段設定。
        ml_state = gr.State(max_length)
        cuda_state = gr.State(no_cuda)

        # ---- 模型載入區（取代啟動時自動載入）----
        with gr.Accordion("模型（請先載入）", open=True):
            gr.Markdown(
                "指定本機檢查點資料夾（內含 `config.json`、`model.safetensors`、`tokenizer.json` 等），"
                "或在下方瀏覽器點選資料夾內任一檔案。"
            )
            with gr.Row():
                checkpoint_path = gr.Textbox(
                    label="檢查點資料夾路徑",
                    value=default_ckpt,
                    placeholder="例如 outputs/Colab/best 或 C:/models/toxicn-best",
                    scale=4,
                )
                load_btn = gr.Button("載入模型", variant="primary", scale=1)
            status = gr.Markdown(
                "ℹ️ 尚未載入模型。" if not default_ckpt
                else f"ℹ️ 已預填路徑：`{default_ckpt}`，請按「載入模型」。"
            )
            with gr.Accordion("瀏覽本機檔案以選取檢查點", open=False):
                browser = gr.FileExplorer(
                    glob="**/config.json",
                    root_dir=browse_root,
                    file_count="single",
                    label=f"於 {browse_root} 下尋找檢查點（顯示各資料夾的 config.json）",
                )

        with gr.Tab("單句偵測"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp = gr.Textbox(
                        label="輸入中文留言",
                        placeholder="例如：你這種人就是社會的垃圾……",
                        lines=4,
                        autofocus=True,
                    )
                    threshold = gr.Slider(
                        0.05, 0.95, value=0.5, step=0.05,
                        label="判定閾值　p(有毒) ≥ 閾值 → 判為有毒",
                    )
                    with gr.Row():
                        btn = gr.Button("偵測", variant="primary")
                        clear = gr.Button("清除")
                with gr.Column(scale=2):
                    verdict = gr.HTML(label="判定")
                    probs = gr.Label(label="信心度", num_top_classes=2)

            gr.Examples(examples=EXAMPLES, inputs=inp, label="點選範例試試看")

            btn.click(classify_single, [inp, threshold], [verdict, probs])
            inp.submit(classify_single, [inp, threshold], [verdict, probs])
            threshold.release(classify_single, [inp, threshold], [verdict, probs])
            clear.click(lambda: ("", {}, ""), None, [inp, probs, verdict])

        with gr.Tab("批次偵測"):
            blob = gr.Textbox(
                label="每行一句，可一次貼上多句",
                lines=8,
                placeholder="第一句…\n第二句…\n第三句…",
            )
            bthr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="判定閾值")
            bbtn = gr.Button("批次偵測", variant="primary")
            bsummary = gr.Markdown()
            btable = gr.Dataframe(
                headers=["文字", "判定", "p(有毒)"],
                datatype=["str", "str", "number"],
                wrap=True,
                label="逐句結果",
            )
            bbtn.click(classify_batch, [blob, bthr], [btable, bsummary])

        # 事件繫結（模型載入 / 瀏覽器選取）。
        load_btn.click(load_model, [checkpoint_path, ml_state, cuda_state], [status])
        checkpoint_path.submit(load_model, [checkpoint_path, ml_state, cuda_state], [status])
        browser.change(on_browse, browser, checkpoint_path)

        gr.Markdown(
            "---\n"
            "⚠️ 模型僅針對「是否有毒」做二分類，結果僅供研究展示；"
            "資料集 ToxiCN 以 CC BY-NC-ND 4.0 授權，禁止商用。"
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="中文毒性言論偵測展示 (Gradio)")
    parser.add_argument("--checkpoint", default=None,
                        help="預先填入輸入框的檢查點路徑（不會自動載入，仍需手動按「載入模型」）")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--browse-root", default=str(PROJECT_ROOT),
                        help="檔案瀏覽器的根目錄（預設為專案目錄）")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="產生公開分享連結")
    args = parser.parse_args()

    # 僅作為輸入框預填，不在啟動時載入任何模型。
    default_ckpt = args.checkpoint or os.environ.get("TOXICN_CKPT", "") or ""

    demo = build_demo(
        default_ckpt=default_ckpt,
        max_length=args.max_length,
        no_cuda=args.no_cuda,
        browse_root=args.browse_root,
    )
    demo.launch(
        theme=gr.themes.Soft(primary_hue="red"),
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
