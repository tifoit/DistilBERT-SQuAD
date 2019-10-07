from model import Model

model = Model('model')

context = "Netflix uses a variety of methods to help you find TV shows and movies to enjoy. You can find TV shows and movies through Recommendations or Search, or by browsing through categories."

question = "How do I find TV shows and movies on Netflix?"

answer = model.predict(context, question)

print("Question: " + question)
print("Answer: " + answer["answer"])
