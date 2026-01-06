# Chatbot Inteligente do IFPI Campus Parnaíba
# Aprendizado de Máquina Supervisionado - Naive Bayes

import unicodedata
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import RSLPStemmer


# PRÉ-PROCESSAMENTO DE TEXTO


stemmer = RSLPStemmer()

stop_words = {
    "a","agora","algum","alguma","aquele","aqueles","onde","de","do","da","em","um","uma",
    "os","as","o","com","como","para","por","que","se","seu","sua","voce","voces",
    "qual","quais","quem","quando","por favor","bom dia","boa tarde","boa noite",
    "gostaria","saber","queria","dizer","informar","explicar","tirar","duvida",
    "pergunta","responder","ajuda","auxilio","favor","gentileza","oi","ola",
    "ifpi","campus","parnaiba","phb","instituto","federal","piaui","piauí",
    "aluno","estudante","curso","escola","colegio","unidade","presencial","remoto"
}

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z\s]", "", texto)

    palavras = texto.split()
    palavras = [stemmer.stem(p) for p in palavras if p not in stop_words]

    return " ".join(palavras)


# BASE DE TREINAMENTO


dados = [

    # 1. Ingresso Técnico
    ("como estudar no ifpi parnaiba", "ingresso_tecnico"),
    ("procedimentos seletivo cursos tecnicos", "ingresso_tecnico"),
    ("quando abre seletivo ensino medio", "ingresso_tecnico"),
    ("onde faço prova ifpi", "ingresso_tecnico"),
    ("documentos para inscricao ifpi", "ingresso_tecnico"),

    # 2. Ingresso Superior
    ("como entro na faculdade ifpi", "ingresso_superior"),
    ("ifpi tem vestibular proprio", "ingresso_superior"),
    ("nota de corte fisica parnaiba", "ingresso_superior"),
    ("preciso fazer enem", "ingresso_superior"),
    ("como funciona sisu ifpi", "ingresso_superior"),

    # 3. Cursos
    ("quais cursos tem ifpi parnaiba", "cursos"),
    ("lista cursos tecnicos superiores", "cursos"),
    ("tem curso informatica", "cursos"),
    ("licenciaturas disponiveis", "cursos"),
    ("ifpi tem engenharia", "cursos"),

    # 4. Assistência Estudantil
    ("como ganhar bolsa ifpi", "assistencia"),
    ("programas assistencia estudantil", "assistencia"),
    ("auxilio transporte", "assistencia"),
    ("auxilio moradia", "assistencia"),
    ("aluno recebe dinheiro", "assistencia"),

    # 5. Localização e Contato
    ("onde fica ifpi parnaiba", "localizacao"),
    ("endereco telefone ifpi", "localizacao"),
    ("email ifpi parnaiba", "localizacao"),
    ("como chegar ifpi", "localizacao"),
    ("horario funcionamento campus", "localizacao"),

    # 6. Sistemas
    ("como ver notas", "sistemas"),
    ("diferenca suap moodle", "sistemas"),
    ("esqueci senha suap", "sistemas"),
    ("material aulas online", "sistemas"),
    ("entrar moodle", "sistemas"),

    # 7. Documentos
    ("declaracao matricula", "documentos"),
    ("emitir diploma", "documentos"),
    ("historico escolar", "documentos"),
    ("protocolo suap documentos", "documentos"),
    ("prazo diploma", "documentos"),

    # 8. Estágio
    ("como conseguir estagio", "estagio"),
    ("documentos estagio obrigatorio", "estagio"),
    ("sala ciec", "estagio"),
    ("vaga emprego aluno", "estagio"),
    ("estagiar qualquer empresa", "estagio"),

    # 9. Biblioteca
    ("renovar livro biblioteca", "biblioteca"),
    ("horario biblioteca", "biblioteca"),
    ("entrar sophia", "biblioteca"),
    ("wifi biblioteca", "biblioteca"),
    ("multa atraso livro", "biblioteca"),

    # 10. Calendário
    ("quando comecam ferias", "calendario"),
    ("consultar calendario academico", "calendario"),
    ("hoje tem aula", "calendario"),
    ("quando termina semestre", "calendario"),
    ("feriado parnaiba", "calendario"),

    # 11. Uniforme
    ("precisa usar farda", "uniforme"),
    ("norma uniforme campus", "uniforme"),
    ("comprar farda", "uniforme"),
    ("aluno faculdade usa uniforme", "uniforme"),
    ("entrar chinelo", "uniforme"),

    # 12. Pesquisa e Extensão
    ("como virar pesquisador", "pesquisa"),
    ("projetos pesquisa extensao", "pesquisa"),
    ("o que e pibic", "pesquisa"),
    ("bolsa extensao", "pesquisa"),
    ("fazer projeto", "pesquisa"),

    # 13. Refeitório
    ("almoco gratis ifpi", "refeitorio"),
    ("cardapio hoje", "refeitorio"),
    ("usar refeitorio", "refeitorio"),

    # 14. Laboratórios
    ("entrar laboratorio bermuda", "laboratorio"),
    ("precisa jaleco", "laboratorio"),
    ("regras seguranca laboratorio", "laboratorio"),

    # 15. Grêmio e CA
    ("falar gremio", "representacao"),
    ("sala ca fisica", "representacao"),
    ("representa alunos campus", "representacao"),

    # 16. Estacionamento
    ("estacionar moto", "estacionamento"),
    ("guardar bicicleta", "estacionamento"),
    ("estacionamento gratuito", "estacionamento")
]

perguntas = [normalizar_texto(p) for p, c in dados]
categorias = [c for p, c in dados]


# RESPOSTAS


respostas = {
    "ingresso_tecnico": "O ingresso nos cursos técnicos ocorre por Exame de Seleção. As inscrições são feitas em certames.ifpi.edu.br.",
    "ingresso_superior": "O ingresso nos cursos superiores ocorre via SISU, utilizando a nota do ENEM.",
    "cursos": "O campus Parnaíba oferece cursos técnicos e superiores, incluindo Informática, Edificações e Licenciaturas.",
    "assistencia": "O IFPI oferece auxílios e bolsas por meio da Assistência Estudantil via SUAP.",
    "localizacao": "O IFPI Parnaíba fica na Av. Capitão Claro, s/n, Centro. Telefone: (86) 3315-6900.",
    "sistemas": "O SUAP é usado para notas e frequência, e o Moodle para aulas online.",
    "documentos": "Documentos e diplomas devem ser solicitados via protocolo no SUAP ou na SEAC.",
    "estagio": "O setor responsável por estágios é a CIEC. O estágio deve ser validado antes do início.",
    "biblioteca": "A biblioteca funciona nos turnos manhã, tarde e noite e usa o sistema Sophia.",
    "calendario": "O calendário acadêmico está disponível no site ifpi.edu.br/parnaiba.",
    "uniforme": "O uniforme é obrigatório no ensino médio integrado. No superior, não é obrigatório.",
    "pesquisa": "O IFPI oferece projetos de pesquisa e extensão, como PIBIC e PIBITI.",
    "refeitorio": "O campus oferece alimentação escolar pelo PNAE conforme critérios específicos.",
    "laboratorio": "O uso de EPIs é obrigatório nos laboratórios.",
    "representacao": "A representação estudantil é feita pelo Grêmio e pelos Centros Acadêmicos.",
    "estacionamento": "O campus possui estacionamento gratuito para estudantes."
}


# TREINAMENTO DO MODELO


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(perguntas)

modelo = MultinomialNB()
modelo.fit(X, categorias)

print("🤖 Chatbot IFPI Parnaíba (IA com Machine Learning)")
print("Digite sua pergunta ou 'sair'\n")

while True:
    entrada = input("Você: ")
    if entrada.lower() == "sair":
        print("Chatbot: Até logo!")
        break

    entrada_proc = normalizar_texto(entrada)
    entrada_vec = vectorizer.transform([entrada_proc])
    categoria = modelo.predict(entrada_vec)[0]

    print("Chatbot:", respostas.get(categoria, "Desculpa, não consegui entender."))

