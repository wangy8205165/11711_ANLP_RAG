import json
import os

# ==============================
# 定义五大类及其数据集
# ==============================
categories = {
    "generalinfo": [
        "wikipedia",
        "city_of_pittsburgh",
        "brittanica",
        "visitpittsburgh",
        "tax_regs",
        "about_cmu",
        "operating_budget"
    ],
    "events": [
        "PittEventCalendar",
        "DtPittCalendar",
        "PghCityPaperEvent",
        "CMUEvent",
        "CMUEventCalendar"
    ],
    "musicandculture": [
        "Symphony",
        "Opera",
        "trustarts",
        "carnegiemuseum",
        "heinz_history_center",
        "thefrick"
    ],
    "food": [
        "food_festival",
        "picklesburgh",
        "pghtacofest",
        "pittsburgh_restaurant_week",
        "littleItaly",
        "banana_split_fest"
    ],
    "sports": [
        "sport_pittsburgh",
        "steeler",
        "penguins",
        "pirates"
    ]
}

# ==============================
# 定义路径模板
# ==============================
base_question = "data/test/question_{}.txt"
base_chunk = "data/chunks/chunks_{}.jsonl"
base_ref = "data/reference/reference_{}.json"

# ==============================
# 合并函数
# ==============================
def merge_files(category, datasets):
    print(f"🔹 Merging category: {category}")

    # ---------- 合并 question ----------
    merged_questions = []
    for name in datasets:
        path = base_question.format(name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                merged_questions.extend(line.strip() for line in f if line.strip())
        else:
            print(f"⚠️ Missing question file: {path}")

    output_q = f"data/test/question_{category}.txt"
    os.makedirs(os.path.dirname(output_q), exist_ok=True)
    with open(output_q, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_questions))
    print(f"✅ Saved {output_q} ({len(merged_questions)} lines)")

    # ---------- 合并 chunks ----------
    merged_chunks = []
    for name in datasets:
        path = base_chunk.format(name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "text" in data:
                        merged_chunks.append(data["text"])
        else:
            print(f"⚠️ Missing chunk file: {path}")

    # 重新编号 chunk_id
    new_chunks = []
    for i, text in enumerate(merged_chunks):
        new_chunks.append({
            "chunk_id": f"{i:04d}",  # 从0000开始编号
            "text": text
        })

    output_c = f"data/chunks/chunk_{category}.jsonl"
    os.makedirs(os.path.dirname(output_c), exist_ok=True)
    with open(output_c, "w", encoding="utf-8") as f:
        for c in new_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ Saved {output_c} ({len(new_chunks)} chunks)")

    # ---------- 合并 references ----------
    merged_refs = {}
    idx = 1
    for name in datasets:
        path = base_ref.format(name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                answer = data[0]
                for _, v in answer.items():
                    merged_refs[str(idx)] = v
                    idx += 1
        else:
            print(f"⚠️ Missing reference file: {path}")

    output_r = f"data/reference/reference_{category}.json"
    os.makedirs(os.path.dirname(output_r), exist_ok=True)
    with open(output_r, "w", encoding="utf-8") as f:
        json.dump(merged_refs, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {output_r} ({len(merged_refs)} references)\n")


# ==============================
# 主程序入口
# ==============================
if __name__ == "__main__":
    for cat, dataset_list in categories.items():
        merge_files(cat, dataset_list)

    print("🎉 All merges completed successfully.")
