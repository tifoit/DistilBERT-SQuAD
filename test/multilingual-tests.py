from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch


tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa")
model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa")

qa_pipeline=pipeline('question-answering', model=model, tokenizer=tokenizer)


# alternative using pipelines
def multiple_answer_question(context, question, qa_pipeline):
    contexts = [context] * len(questions)  # creates a list of contexts. There's a context per asked question
    answers = qa_pipeline(question=questions, context=contexts)

    # if a single element is returned, then convert it to list
    if not isinstance(answers, list):
        answers = [answers]

    for i in range(len(answers)):
        print(i)
        print("Q:", questions[i])
        print(answers[i])
        # print(f"A: {answers[i]['answer']} (score: {answers[i]['score']})")
        print("---")

context = "The US has passed the peak on new coronavirus cases, " \
            "President Donald Trump said and predicted that some states would reopen this month. " \
            "The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world."

questions = ["'How many deaths are there?", "What did president Trump predict?"]

multiple_answer_question(context,questions,qa_pipeline)

print("-----------------------------------------")

context = "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1)," \
    "72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10)."\
    "A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final." \
    "Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds. "\
    "The minimum number of games needed to win, depending on which round a team enters the competition, ranges from six to fourteen."

questions = ["Who can play the FA Cup?",
             "How many wins are required to win the cup?",
             "How many teams played in the FA Cup in 2012?",
             "How many rounds are there in the FA Cup?"]

multiple_answer_question(context,questions,qa_pipeline)


print("-----------------------------------------")


context = "Como dramaturgo, Chéjov se encontraba en el naturalismo, aunque contaba con ciertos toques del simbolismo." \
    "Sus piezas teatrales más conocidas son La gaviota (1896), Tío Vania (1897), Las tres hermanas (1901) y El jardín de los cerezos (1904). " \
    "En ellas Chéjov ideó una nueva técnica dramática que él llamó «de acción indirecta», fundada en la insistencia en los detalles de caracterización e interacción " \
    "entre los personajes más que el argumento o la acción directa, " \
    "de forma que en sus obras muchos acontecimientos dramáticos importantes tienen lugar fuera de la escena " \
    "y lo que se deja sin decir muchas veces es más importante que lo que los personajes dicen y expresan realmente."

questions = ["Donde se encontraba Chejov?",
             "Cuales son las obras teatrales mas conocidas de Chejov?",
             "Que es la acción indirecta?",
             "Cual técnica dramática usaba Chejov?",
             "Que es mas importante que lo que dicen los personajes?"]

multiple_answer_question(context,questions,qa_pipeline)

print("-----------------------------------------")


context = "En junio de 1955 la UEFA aprobó organizar una competición entre clubes europeos denominada como Copa de Clubes Campeones Europeos "\
    "(nombre original en francés, Coupe des Clubs Champions Européens) —más conocida como Copa de Europa—."\
    "Esta fue impulsada por el periódico deportivo francés L'Équipe de mano de su director en la época Gabriel Hanot junto con su colega Jacques Ferran,"\
    "y con el apoyo del presidente del Real Madrid Club de Fútbol, Santiago Bernabéu, así como Gusztáv Sebes,"\
    "subsecretario de deportes de Hungría y vicepresidente de la UEFA"

questions = ["Que es la Copa de Europa?",
             "Cuando se fundo la Copa de Campeones?",
             "Quien era Santiago Bernabeu?",
             "Quien es el director de  L'Equipe?",
             "Quien promovio la creacion de la Copa de Europa?",
             "Quien era el presidente de la UEFA cuando se fundo la Copa de Campeones?",
             "Quien era el vicepresidente de la UEFA cuando se fundo la Copa de Campeones?"]

multiple_answer_question(context,questions,qa_pipeline)


'''
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="pt")
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])

outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
'''
