import json
from pathlib import Path

import pandas as pd


BASE = Path(r"E:\project\xiaozhi\Uni-Sign-storage\out")
MODELS = {
    "pose-only": BASE / "ablation_pose_only" / "test100_eval.csv",
    "rgb-support": BASE / "ablation_rgb_support" / "test100_eval.csv",
    "full-model": BASE / "ablation_uncert_gate" / "test100_eval.csv",
}

OUT_JSON = Path(r"E:\project\xiaozhi\Uni-Sign\out\paper_metrics\sentence_slicing_test100_summary.json")
OUT_MD = Path(r"E:\project\xiaozhi\Uni-Sign\docs\04_论文稿件\2026-04-25_论文定稿材料\04_句型切片正式表.md")


PUNCT = "".join(
    [
        "\uFF0C",  # ，
        "\u3002",  # 。
        "\uFF1F",  # ？
        "\uFF01",  # ！
        "\uFF1B",  # ；
        "\uFF1A",  # ：
        "\u3001",  # 、
        "\u201C",
        "\u201D",
        "\u2018",
        "\u2019",
        "\uFF08",
        "\uFF09",
        "(",
        ")",
        " ",
    ]
)

TIME_TOKENS = [
    "\u5E74",  # 年
    "\u6708",  # 月
    "\u65E5",  # 日
    "\u53F7",  # 号
    "\u70B9",  # 点
    "\u661F\u671F",  # 星期
    "\u5468",  # 周
    "\u793C\u62DC",  # 礼拜
    "\u4ECA\u5929",  # 今天
    "\u6628\u5929",  # 昨天
    "\u660E\u5929",  # 明天
    "\u65E9\u4E0A",  # 早上
    "\u665A\u4E0A",  # 晚上
    "\u4E2D\u5348",  # 中午
    "\u4E0A\u5348",  # 上午
    "\u4E0B\u5348",  # 下午
    "\u65F6\u5019",  # 时候
    "\u5C0F\u65F6",  # 小时
    "\u5206\u949F",  # 分钟
    "\u79D2",  # 秒
]

QUESTION_TOKENS = [
    "\uFF1F",  # ？
    "?",
    "\u5417",  # 吗
    "\u5462",  # 呢
    "\u591A\u5C11",  # 多少
    "\u4EC0\u4E48",  # 什么
    "\u4E3A\u4EC0\u4E48",  # 为什么
    "\u4E3A\u4F55",  # 为何
    "\u600E\u4E48",  # 怎么
    "\u600E\u6837",  # 怎样
    "\u54EA",  # 哪
    "\u54EA\u91CC",  # 哪里
    "\u54EA\u513F",  # 哪儿
    "\u8C01",  # 谁
    "\u51E0",  # 几
]

NEG_TOKENS = [
    "\u4E0D",  # 不
    "\u6CA1",  # 没
    "\u65E0",  # 无
    "\u4E0D\u80FD",  # 不能
    "\u4E0D\u4F1A",  # 不会
    "\u4E0D\u662F",  # 不是
    "\u6CA1\u6709",  # 没有
    "\u522B",  # 别
    "\u52FF",  # 勿
]

NUM_TOKENS = list("0123456789") + list(
    [
        "\u4E00",
        "\u4E8C",
        "\u4E09",
        "\u56DB",
        "\u4E94",
        "\u516D",
        "\u4E03",
        "\u516B",
        "\u4E5D",
        "\u5341",
        "\u767E",
        "\u5343",
        "\u4E07",
        "\u4E24",
        "\u96F6",
        "\u3007",
    ]
)


def clean_len(text: str) -> int:
    t = str(text)
    for ch in PUNCT:
        t = t.replace(ch, "")
    return len(t)


def contains_any(text: str, tokens: list[str]) -> bool:
    t = str(text)
    return any(tok in t for tok in tokens)


def build_summary() -> list[dict]:
    frames = {name: pd.read_csv(path) for name, path in MODELS.items()}
    ref_df = frames["pose-only"][["dataset_index", "reference"]].copy()

    ref_df["char_len"] = ref_df["reference"].apply(clean_len)
    ref_df["is_long"] = ref_df["char_len"] >= 10
    ref_df["is_short"] = ref_df["char_len"] < 10
    ref_df["has_time"] = ref_df["reference"].apply(lambda x: contains_any(x, TIME_TOKENS))
    ref_df["is_question"] = ref_df["reference"].apply(lambda x: contains_any(x, QUESTION_TOKENS))
    ref_df["has_negation"] = ref_df["reference"].apply(lambda x: contains_any(x, NEG_TOKENS))
    ref_df["has_number"] = ref_df["reference"].apply(lambda x: contains_any(x, NUM_TOKENS))

    categories = [
        ("\u957F\u53E5", "is_long"),  # 长句
        ("\u77ED\u53E5", "is_short"),  # 短句
        ("\u65F6\u95F4\u8868\u8FBE", "has_time"),  # 时间表达
        ("\u95EE\u53E5", "is_question"),  # 问句
        ("\u5426\u5B9A", "has_negation"),  # 否定
        ("\u6570\u5B57", "has_number"),  # 数字
    ]

    summary: list[dict] = []
    for cat_name, col in categories:
        idxs = set(ref_df.loc[ref_df[col], "dataset_index"].tolist())
        row = {"category": cat_name, "count": len(idxs)}
        for model_name, df in frames.items():
            sub = df[df["dataset_index"].isin(idxs)]
            row[f"{model_name}_avgsim"] = round(float(sub["similarity"].mean()), 4) if len(sub) else 0.0

        vals = {
            "pose-only": row["pose-only_avgsim"],
            "rgb-support": row["rgb-support_avgsim"],
            "full-model": row["full-model_avgsim"],
        }
        best = max(vals, key=vals.get)
        if best == "full-model":
            note = "reliability-aware \u6700\u503C\u5F97\u4FDD\u7559"
        elif best == "pose-only":
            note = "pose \u66F4\u7A33"
        else:
            note = "RGB \u6709\u6761\u4EF6\u589E\u76CA"
        row["conclusion"] = note
        summary.append(row)
    return summary


def write_outputs(summary: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# \u53E5\u578B\u5207\u7247\u6B63\u5F0F\u8868",
        "",
        "\u6C47\u603B\u6587\u4EF6\uFF1A",
        "",
        "1. [sentence_slicing_test100_summary.json](/abs/path/E:/project/xiaozhi/Uni-Sign/out/paper_metrics/sentence_slicing_test100_summary.json)",
        "",
        "\u56FA\u5B9A\u5BF9\u7167\u7EC4\uFF1A",
        "",
        "1. `ablation_pose_only`",
        "2. `ablation_rgb_support`",
        "3. `ablation_uncert_gate`",
        "",
        "| Category | Count | pose-only AvgSim | rgb-support AvgSim | full-model AvgSim | \u7ED3\u8BBA |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in summary:
        lines.append(
            f"| {row['category']} | {row['count']} | {row['pose-only_avgsim']:.4f} | {row['rgb-support_avgsim']:.4f} | {row['full-model_avgsim']:.4f} | {row['conclusion']} |"
        )

    lines += [
        "",
        "\u5F53\u524D\u5207\u7247\u89C4\u5219\uFF1A",
        "",
        "1. `\u957F\u53E5`\uFF1A\u53BB\u6807\u70B9\u540E\u7684\u53C2\u8003\u6587\u672C\u957F\u5EA6 `>= 10`",
        "2. `\u77ED\u53E5`\uFF1A\u53BB\u6807\u70B9\u540E\u7684\u53C2\u8003\u6587\u672C\u957F\u5EA6 `< 10`",
        "3. `\u65F6\u95F4\u8868\u8FBE`\uFF1A\u542B `\u5E74/\u6708/\u65E5/\u53F7/\u70B9/\u661F\u671F/\u5468/\u4ECA\u5929/\u6628\u5929/\u660E\u5929/\u65E9\u4E0A/\u665A\u4E0A/\u4E2D\u5348/\u4E0A\u5348/\u4E0B\u5348` \u7B49",
        "4. `\u95EE\u53E5`\uFF1A\u542B `\uFF1F/?/\u5417/\u5462/\u591A\u5C11/\u4EC0\u4E48/\u4E3A\u4EC0\u4E48/\u600E\u4E48/\u8C01/\u51E0` \u7B49",
        "5. `\u5426\u5B9A`\uFF1A\u542B `\u4E0D/\u6CA1/\u65E0/\u4E0D\u80FD/\u4E0D\u4F1A/\u4E0D\u662F/\u6CA1\u6709/\u522B/\u52FF` \u7B49",
        "6. `\u6570\u5B57`\uFF1A\u542B\u963F\u62C9\u4F2F\u6570\u5B57\u6216\u4E2D\u6587\u6570\u5B57\u5B57\u7B26",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = build_summary()
    write_outputs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
