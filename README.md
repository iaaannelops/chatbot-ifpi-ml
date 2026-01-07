# Chatbot Inteligente para Perguntas Frequentes – IFPI
Este projeto consiste no desenvolvimento de um chatbot inteligente capaz de responder automaticamente perguntas frequentes sobre o Instituto Federal do Piauí (IFPI).

O sistema utiliza técnicas de Aprendizado de Máquina Supervisionado para classificar perguntas feitas pelos usuários e fornecer respostas adequadas, mesmo quando a pergunta não é exatamente igual às utilizadas no treinamento.

## Objetivo
Desenvolver um chatbot utilizando aprendizado de máquina para:

Classificar automaticamente perguntas frequentes

Simular um atendimento automatizado

Aplicar conceitos básicos de Inteligência Artificial

## Inteligência Artificial Utilizada
O projeto utiliza Aprendizado de Máquina Supervisionado, com foco em classificação de texto.

O algoritmo escolhido foi o Naive Bayes, que aprende padrões a partir de exemplos de perguntas previamente classificadas em categorias.

Mesmo com variações na forma das perguntas, o modelo consegue identificar o assunto e retornar a resposta correta.

## Tecnologias Utilizadas
Linguagem: Python

Biblioteca de IA: NLTK e Scikit-learn

Editor de código: VS Code

Versionamento: Git e GitHub

Organização do projeto: Trello

## Funcionamento do Sistema
O chatbot segue o seguinte fluxo:ㅤㅤ
ㅤㅤ
Pergunta do usuárioㅤㅤ
↓ㅤㅤ
Pré-processamento do texto (limpeza, remoção de stop words e stemming)ㅤㅤ
↓ㅤㅤ
Vectorização com CountVectorizerㅤㅤ
↓ㅤㅤ
Classificação com Naive Bayesㅤㅤ
↓ㅤㅤ
Resposta automática baseada na categoriaㅤㅤ
ㅤㅤ
## Exemplos de Uso
Pergunta: Como faço a inscrição para o ensino médio do IFPI?
Resposta: O ingresso nos cursos técnicos do IFPI ocorre por meio de Exame de Seleção.

Pergunta: O IFPI cobra mensalidade?
Resposta: O IFPI é uma instituição pública e gratuita.

## Dificuldades Encontradas
Entendimento inicial dos conceitos de aprendizado de máquina

Instalação e configuração das bibliotecas

Organização da base de treinamento

Ajustes no pré-processamento para melhorar a precisão

## Possíveis Melhorias
Aumentar a base de treinamento

Adicionar novas categorias de perguntas

Criar uma interface gráfica

Integrar o chatbot a uma página web

## Equipe
Ianne – Desenvolvimento do código e modelo de IA

Davi – Documentação e apoio na apresentação

Raíssa – Organização da base de perguntas

Nikole – Organização do Trello e testes

Nathielly – Apresentação e revisão do projeto

## Referências
Welcome to Python.org

scikit-learn: machine learning in Python — scikit-learn 1.8.0 documentation

NLTK :: Natural Language Toolkit

Material da disciplina de Inteligência Artificial
