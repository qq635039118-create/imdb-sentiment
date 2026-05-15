import pandas as pd
import random

# 模拟 IMDB 评论数据
positive_words = ["great", "amazing", "excellent", "wonderful", "fantastic", "love", "best", "beautiful", "perfect", "awesome"]
negative_words = ["terrible", "awful", "bad", "worst", "boring", "hate", "disappointing", "poor", "waste", "horrible"]
neutral_words = ["movie", "film", "story", "acting", "plot", "character", "scene", "director", "watch", "see"]

def generate_review(label):
    words = []
    if label == 1:
        words.extend(random.sample(positive_words, 3))
        words.extend(random.sample(neutral_words, 4))
        words.extend(random.sample(positive_words, 2))
    else:
        words.extend(random.sample(negative_words, 3))
        words.extend(random.sample(neutral_words, 4))
        words.extend(random.sample(negative_words, 2))
    random.shuffle(words)
    return " ".join(words)

# 生成 500 条数据
data = []
for i in range(500):
    label = random.randint(0, 1)
    review = generate_review(label)
    data.append({"review": review, "sentiment": label})

df = pd.DataFrame(data)
df.to_csv("imdb_top_500.csv", index=False)
print(f"Generated {len(df)} samples")
print(df.head())
