from datasets import load_dataset
import pandas as pd

def filter_datasets(datasets):
  qa_data = []
  for dataset in datasets:
    chosen_text = dataset["chosen"]
    rejected_text = dataset["rejected"]
    # 単一プロンプトのみに限定
    if chosen_text.count("Human:") == 1:
      qa = (chosen_text.split("\n\nHuman:")[1], rejected_text.split("\n\nHuman:")[1])
      qa = ((str(qa[0])).split("\n\nAssistant:"),(str(qa[1])).split("\n\nAssistant:") )


      qa_data.append(
          {
          "question": qa[0][0],
          "acceptedAnswer": qa[0][1],
          "rejectedAnswer": qa[1][1]
          }
      )

  return qa_data

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
qa_data = filter_datasets(dataset)

df = pd.DataFrame(qa_data)
df["id"] = range(len(df))
df = df[
    ["id","question","acceptedAnswer","rejectedAnswer"]
]

# 学習用のjsonl
output_filename = "hh-rfhf_for_dpo.jsonl"
df.to_json(
    output_filename,
    orient='records',
    lines=True,
    force_ascii=False
)

# 確認用のjson
output_filename_json = "hh-rfhf_for_dpo.json"
df.to_json(
    output_filename_json,
    orient='records',
    indent=2,
    force_ascii=False
)
