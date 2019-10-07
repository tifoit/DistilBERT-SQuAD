from model import Model

model = Model('model')

context = "We deliver every day from morning until late at night, and different restaurants will have different opening times. Visit the homepage or the app to see which restaurants are available in your area."

question = "When can I order?"

answer = model.predict(context, question)

print("Question: " + question)
print("Answer: " + answer["answer"])
