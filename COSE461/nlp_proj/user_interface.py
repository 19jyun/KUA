from transformers import AutoModelForSequenceClassification, RobertaTokenizer
from torch.nn.functional import softmax
import pandas as pd

#Last version before merging into multimodal

# DataFrame을 초기화합니다.
df = pd.DataFrame(columns=['input', 'probability'])

# 토크나이저 로드
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained("trained_model")

while True:
    # 사용자로부터 입력 받기
    sentence = input("Enter a sentence: ")

    # 'Exit'를 입력하면 루프를 종료합니다.
    if sentence.lower() == 'exit' or sentence == '0':
        break

    # 문장을 모델의 입력 형식에 맞게 변환
    inputs = tokenizer(sentence, return_tensors="pt")

    # 예측 수행
    outputs = model(**inputs)

    # 예측 결과를 확률로 변환
    probs = softmax(outputs.logits, dim=-1)

    # 풍자일 확률을 출력하고 DataFrame에 추가합니다.
    prob = probs[0][1].item()
    print("Satire probability:", prob)
    df.loc[len(df)] = [sentence, prob]

# DataFrame을 CSV 파일에 저장합니다.
df.to_csv('results/inputs_and_probs.csv', index=False)