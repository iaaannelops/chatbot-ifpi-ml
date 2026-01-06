# Chatbot Inteligente do IFPI com Aprendizado de Máquina
# Classificação de texto usando Naive Bayes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Base de treinamento (perguntas + categorias)
perguntas = [
    "o que é o ifpi",
    "o ifpi é público",
    "o ifpi é gratuito",

    "quais cursos o ifpi oferece",
    "o ifpi tem curso técnico",
    "o ifpi oferece ensino médio integrado",

    "como entrar no ifpi",
    "tem processo seletivo",
    "o ifpi usa o enem",

    "qual o horário de funcionamento",
    "o ifpi funciona à noite",

    "como entrar em contato com o ifpi",
    "onde fica o ifpi"
]

categorias = [
    "instituicao",
    "instituicao",
    "instituicao",

    "cursos",
    "cursos",
    "cursos",

    "ingresso",
    "ingresso",
    "ingresso",

    "horario",
    "horario",

    "contato",
    "contato"
]

# Respostas por categoria
respostas = {
    "instituicao": "O IFPI é uma instituição pública e gratuita de ensino.",
    "cursos": "O IFPI oferece cursos técnicos, ensino médio integrado e cursos superiores.",
    "ingresso": "O ingresso no IFPI ocorre por meio de processos seletivos e, em alguns casos, pelo Enem.",
    "horario": "O IFPI funciona, geralmente, das 8h às 18h e também possui cursos noturnos.",
    "contato": "O contato pode ser feito pelo site oficial do IFPI ou pela secretaria do campus."
}

# Transformar texto em números
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(perguntas)

# Treinar o modelo
modelo = MultinomialNB()
modelo.fit(X, categorias)

print("🤖 Chatbot Inteligente do IFPI")
print("Digite sua pergunta ou 'sair' para encerrar.\n")

while True:
    entrada = input("Você: ").lower()

    if entrada == "sair":
        print("Chatbot: Até logo!")
        break

    entrada_transformada = vectorizer.transform([entrada])
    categoria_prevista = modelo.predict(entrada_transformada)[0]

    print("Chatbot:", respostas[categoria_prevista])
