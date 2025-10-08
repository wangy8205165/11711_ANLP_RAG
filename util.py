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

if __name__ == "__main__":
    input_path = "data/chunks/chunks_carnegie_museum.jsonl"
    output_path = "data/chunks/chunks_carnegie_museum_2.jsonl"
    renumber_jsonl(input_path, output_path, width=3, prefix="")
    print("处理完毕，总行数：", sum(1 for _ in open(output_path, 'r', encoding='utf-8')))
