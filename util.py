import json

def renumber_jsonl(input_path, output_path, width=3, prefix=""):
    """
    将 input_path 路径的 jsonl 文件中，每一行 {"chunk_id": ..., "text": ...}
    重命名 chunk_id，从 0 开始按出现顺序编号。
    
    参数：
    - input_path: 原始 jsonl 文件路径
    - output_path: 写重编号后文件的路径
    - width: 宽度（编号字符串用零填充至该长度），如 width=3，则编号依次为 "000","001","002"
    - prefix: 给编号加前缀，比如 prefix="chunk_"，则编号为 "chunk_000" 等
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        counter = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过无效 JSON 行: {line!r} — 错误: {e}")
                continue
            
            # 生成新的 chunk_id
            new_id = f"{prefix}{counter:0{width}d}"
            obj["chunk_id"] = new_id
            
            # 写出
            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")
            
            counter += 1

"data/reference/reference_trustart.txt"
"data/reference/reference_trustarts.json"


def generate_reference(txt_path, json_path):
    txt_path = txt_path
    json_path = json_path

    # 读取 txt 文件
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # 去掉空行与换行符

    # 生成字典，键为序号（从1开始），值为对应行文字
    data = {str(i + 1): line for i, line in enumerate(lines)}

    # 写入 JSON 文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([data], f, ensure_ascii=False, indent=4)

    print("✅ 已成功生成 JSON 文件:", json_path)


if __name__ == "__main__":


    input_path = "data/chunks/chunks_thefrick_2.jsonl"
    output_path = "data/chunks/chunks_thefrick.jsonl"
    renumber_jsonl(input_path, output_path, width=3, prefix="")
    print("处理完毕，总行数：", sum(1 for _ in open(output_path, 'r', encoding='utf-8')))

    
    txt_path = "data/reference/reference_thefrick.txt"
    json_path = "data/reference/reference_thefrick.json"
    generate_reference(txt_path,json_path)
