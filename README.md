# Aula: Desenvolvimento de Agentes de IA - de 0 a 100

Este repositório contém o conteúdo detalhado da aula sobre o desenvolvimento de agentes de IA, cobrindo desde a preparação do ambiente até conceitos avançados como RAG, MCP e LangGraph.

---

## Preparando o ambiente

1. Instalando os pacotes
2. Realizando os imports
3. Definindo as configurações (Variáveis de ambiente ou Secrets)
4. Baixando os arquivos de suporte.

## Instalando os **pacotes**

```python
# Instalando todos os pacotes que vamos utilizar no curso.
!pip install python-dotenv openai ipykernel numexpr \
             mcp>=1.8.0 langchain langchain_groq langchain_google_genai \
             langchain_openai langchain_experimental langchain-community langchain_groq\
             langgraph pypdf chromadb langchain_chroma fastmcp langchain_mcp_adapters==0.0.9 \
             pydantic graphviz grandalf pydot requests==2.32.4 matplotlib google-generativeai pandas tabulate -q
```

```python
# Linux
# Instalando o graphviz, para gerarmos nossos fluxos em PNG
!apt install graphviz graphviz-dev -y
```

```python
# Windows
# Download: https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/11.0.0/windows_10_cmake_Release_graphviz-install-11.0.0-win64.exe
# Instalar o .exe
# Adicionar no PATH ==> "C:\Program Files\Graphviz\bin"

# link oficial de download do instalador
# https://graphviz.org/download/
```

```python
# Linux
# Instalando o pacote do graphviz no python
!pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-I/usr/include/graphviz" pygraphviz -q
```

```python
# Windows
#!pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz
```

## Importando os pacotes

```python
# =======================
# Bibliotecas da standard library (Python)
# =======================

# Sistema operacional: criação de diretórios, configuração e leitura de variáveis de ambiente
import os

# Informações e manipulação da execução do interpretador Python
import sys

# Expressões regulares
import re

# Manipulação de datas e horários
import datetime

# Execução de funções assíncronas
import asyncio

# Conexões seguras e cliente HTTP assíncrono
import ssl
import httpx

# Manipulação de arquivos e diretórios de forma independente do sistema operacional
from pathlib import Path

# Identificadores únicos universais (UUID)
from uuid import uuid4

# Manipulação de CSV
import pandas as pd

# Tipagem estática (anotações e tipos auxiliares)
from typing import Any, List, Union, TypedDict

# Manipulação de XML
from xml.etree import ElementTree as ET

# Envio de e-mails (SMTP e MIME)
from smtplib import SMTP, SMTP_SSL
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
try:
    from email import encoders
except ImportError:
    from email import Encoders as encoders

# Leitura de variáveis de ambiente a partir de arquivos `.env`
from dotenv import load_dotenv

# Requisições HTTP (síncronas)
import requests

# Exibição de gráficos e imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =======================
# OpenAI
# =======================

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# =======================
# Gemini
# =======================

import google.generativeai as gemini

# =======================
# LangChain
# =======================

# Configuração de debug do LangChain
from langchain.globals import set_debug

# Modelos LLM (Large Language Models)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

# Construção de prompts
from langchain.schema import HumanMessage
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Parsers de saída
from langchain.schema.output_parser import StrOutputParser

# Execução de fluxos (Runnables)
from langchain_core.runnables import RunnableLambda

# Criação e execução de agentes
from langchain.agents import (
    Tool,
    AgentExecutor,
    create_tool_calling_agent,
    create_react_agent
)

# Ferramentas customizadas para agentes
from langchain.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools.python.tool import PythonAstREPLTool

# Tipos de memória utilizados em agentes
from langchain.memory import ConversationBufferMemory

# Componentes de RAG (Retrieval-Augmented Generation)
from langchain_chroma import Chroma  # Armazenamento vetorial
from langchain_openai.embeddings import OpenAIEmbeddings  # Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Separador de texto
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Leitura de documentos PDF

# Acesso ao hub LangChain de prompts prontos (https://smith.langchain.com/hub)
from langchain import hub

# =======================
# MCP (Model Context Protocol)
# =======================

# Servidor MCP
from mcp.server.fastmcp import FastMCP

# Cliente MCP multi-servidor
from langchain_mcp_adapters.client import MultiServerMCPClient

# =======================
# LangGraph
# =======================

# Criação de agentes com LangGraph
from langgraph.prebuilt import create_react_agent as create_react_agent_graph

# Sistema de checkpoint em memória
from langgraph.checkpoint.memory import InMemorySaver

# Definição e execução de grafos
from langgraph.graph import StateGraph, END

# =======================
# Outros
# =======================

# Ignora avisos durante a execução
from IPython import get_ipython

import warnings
warnings.filterwarnings('ignore')
```

```python
# Verifica onde o notebook está rodando
RUNNING_IN_COLAB = 'google.colab' in str(get_ipython())
try:
  from google.colab import userdata
except:
  pass

OUTPUT_DOCUMENTS_DIR:str = './documentos/' if not RUNNING_IN_COLAB else '/content/documentos/'

if not RUNNING_IN_COLAB and sys.platform.lower() == "win32" and "Graphviz" not in os.environ["PATH"]: # Somente no Windows
    os.environ["PATH"] = os.getenv("PATH", "") + ";C:\\Program Files\\Graphviz\\bin"
```

## Configurações

```python
# Arquivo de environment (se estiver local)
# Exemplo de um arquivo .env
# ==============================================
# GROQ_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# GOOGLE_API_KEY=
#
# SMTP_USERNAME=
# SMTP_PASSWORD=
# ==============================================

# Caminho de onde foi criado o .env
ENV_PATH:str = '.env'

# No colab utilizamos o <Secrets> Menu esquerda (chave)

def carrega_variaveis_ambiente() -> None:

    # Modo local
    if os.path.exists(ENV_PATH) and not RUNNING_IN_COLAB:
        load_dotenv(ENV_PATH, override=True)

    # Modo Colab
    if RUNNING_IN_COLAB:
      os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
      os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
      #os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
      os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

      os.environ['SMTP_USERNAME'] = userdata.get('SMTP_USERNAME')
      os.environ['SMTP_PASSWORD'] = userdata.get('SMTP_PASSWORD')

carrega_variaveis_ambiente()
```

## Baixando os arquivo para utilizarmos na Aula.

```python
def download(url:str, output_dir:str=OUTPUT_DOCUMENTS_DIR) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'wb') as file:
                file.write(response.content)
            print(f"Arquivo baixado com sucesso: {filepath}")
        else:
            print(f"Erro ao baixar o arquivo. Código de status: {response.status_code}")

```

```python
download('https://middleware.datah.ai/RAG-DATA H.pdf')
```

# Agentes

## Objetivo

1. Entender o que é um agente
2. Conhecer o ciclo de percepção-decisão-ação
3. Distinguir agentes de LLMs simples
4. Criar agentes com Langchain (LCEL / AgentExecutor)
5. Conceitos sobre LLM (Temperatura, Top_p e Min_p)
6. Trabalhar com Ferramentas
7. Trabalhar com Memória
8. Trabalhar com Contexto
9. Prompt
10. Trabalhar com RAG
11. Tipos de Agentes (AgentExcutor / React)
12. Trabalhar com MCP
13. Trabalhar com LangGraph






## O que é um Agente?

Um agente é qualquer **entidade** que pode:
* **Perceber** seu ambiente (ex: através de sensores)
* **Processar** essa percepção
* **Agir** sobre o ambiente (através de atuadores).

![agente](https://middleware.datah.ai/agent_figura_11.png)

## Principais Conceitos

* **Percepção** (Percept): É a informação que o agente coleta do seu ambiente. Para um carro autônomo, por exemplo, a percepção são os dados das câmeras, radares e GPS.

* **Ação** (Action): É o que o agente faz para interagir com o ambiente. No carro autônomo, as ações seriam acelerar, frear ou virar o volante.

* **Função do Agente** (Agent Function): É o "**cérebro**" do agente. É a função que mapeia a sequência de percepções para uma ação. Teoricamente, **como um agente deveria agir** em resposta a uma sequência completa de percepções. Na prática, essa função é implementada por um **programa de agente**.

$$f:P^* \rightarrow A$$

* **Programa do Agente** (Agent Program): É a implementação concreta da função do agente. Existem diferentes tipos de programas de agente, cada um com um nível de complexidade e "inteligência.

$$ f $$

* **Atuadores** (actuators): Executam essa ação no mundo físico. Eles são a parte do agente que interage fisicamente com o ambiente.

## Estrutura para criar um Agente

![agente](https://middleware.datah.ai/agent_figura_01.png)


## Criando um agente utilizando a API da OpenAI

LLM + Prompt


***IMPORTANTE: Este exemplo precisa de uma chave da OpenAI.***


```python
# exemplo_01.py

# Criando o client para trabalharmos com os agentes.
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# ou

client = OpenAI()

# Criando o agente
def agente_manual(pergunta:str) -> str:
    model:str = "gpt-4o-mini"
    prompt:str = f"""
Você é um assistente inteligente com acesso a duas ferramentas:
1. Calculadora
2. Wikipedia

Dado a pergunta abaixo, diga o que pretende fazer.

Pergunta: {pergunta}

Responda no formato:
Ação: [Calculadora|Wikipedia|Responder diretamente]
Motivo: ...
    """

    resposta:ChatCompletion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return resposta.choices[0].message.content
```

```python
# Utilizando nosso primeiro agente
resposta = agente_manual("Qual a raiz quadrada de 256?")
print("Resposta OpenAI:",resposta)
```

```python
resposta = agente_manual("Qual a população de Ribeirão Preto.")
print("Resposta OpenAI:", resposta)
```

## Criando um agente utilizando a API do Gemini

LLM + Prompt


***IMPORTANTE: Este exemplo precisa de uma chave do Gemini.***

```python
# exemplo_01_1.py

# Criando o client para trabalharmos com os agentes.
gemini.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Criando o agente
def agente_manual_gemini(pergunta: str) -> str:
    model_name: str = "gemini-1.5-flash"
    prompt: str = f"""
Você é um assistente inteligente com acesso a duas ferramentas:
1. Calculadora
2. Wikipedia

Dado a pergunta abaixo, diga o que pretende fazer.

Pergunta: {pergunta}

Responda no formato:
Ação: [Calculadora|Wikipedia|Responder diretamente]
Motivo: ...
    """

    # Inicializa o modelo
    model = gemini.GenerativeModel(model_name)
    response = model.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [prompt]
            }
        ]
    )

    return response.text
```

```python
resposta = agente_manual_gemini("Qual a raiz quadrada de 256?")
print("Resposta Gemini:", resposta)
```

```python
resposta = agente_manual_gemini("Qual a população de Ribeirão Preto.")
print("Resposta Gemini:", resposta)
```

## LangChain

O LangChain é um framework em **Python (e JS)** criado para construção de aplicações que usam LLMs (Large Language Models) como ChatGPT, Claude, Mistral, Llama, etc.


**Porque utilizar um Framework?**

* Ele facilita a criação de pipelines, chatbots, assistentes, agentes, RAGs (Retrieval-Augmented Generation), entre outros.

* Facilita a portabilidade entre as LLMs. (Cada LLM tem a sua API com as sua particularidade).

* Ele tem componentes prontos e bem separados (LLMs, Memory, Tools, Chains, Agents).

* Facilita a construção de pipelines complexas.

* Já vem com recursos de tracking, observability, serialization, etc.

* Você pode usar só as partes que quiser (não é obrigatório usar tudo).

Para mais detalhes acesse: https://www.langchain.com/

## Iniciando com o Framework LangChain (LCEL)

### O que é o LangChain Expression Language (LCEL)?

 **LCEL (LangChain Expression Language)** é uma ferramenta poderosa do LangChain projetada para facilitar a construção de cadeias de chamadas (chains) de forma fluida e eficiente. Pense nele como a "cola" que une diferentes componentes de um aplicativo de IA, como modelos de linguagem (LLMs), prompts e ferramentas, em um fluxo de trabalho coerente.

A grande **vantagem do LCEL** é sua capacidade de permitir que os desenvolvedores criem pipelines complexos de maneira simples e declarativa, usando o operador | (pipe), semelhante ao que se usa em shells como Bash. Isso torna a leitura e a escrita das cadeias muito mais intuitiva.

O LCEL não é apenas uma sintaxe elegante; ele traz consigo uma série de benefícios importantes:

* **Streaming**: Ele suporta o streaming de tokens, ou seja, as respostas são geradas em tempo real, em vez de esperar a conclusão total da cadeia. Isso melhora a experiência do usuário, pois a resposta começa a aparecer imediatamente.

* **Paralelismo**: O LCEL executa operações que não dependem umas das outras em paralelo automaticamente, o que melhora o desempenho da sua aplicação.

* **Fallback**: Ele permite a definição de mecanismos de "fallback", onde você pode configurar um plano B caso um modelo ou ferramenta falhe, aumentando a robustez da sua aplicação.

* **Composição**: A facilidade de combinar e reutilizar diferentes partes da sua cadeia, tornando o código mais modular e fácil de manter.

* **Acessibilidade**: Suporte para chamadas síncronas e assíncronas, permitindo que você adapte o código ao seu ambiente.

### Como o LCEL funciona?
O funcionamento do LCEL é baseado no encadeamento de objetos que implementam a interface Runnable. Cada componente do LangChain que pode ser parte de uma cadeia – como PromptTemplate, ChatModel, OutputParser – é um Runnable.

A sintaxe principal é o operador |. Quando você escreve, por exemplo, `prompt | model`, o que está acontecendo por baixo dos panos é o seguinte:

* **Entrada**: A cadeia recebe uma entrada (um dicionário, uma string, etc.).

* **prompt**: A entrada é processada pelo prompt. Por exemplo, uma string é formatada em um PromptValue.

* **model**: O resultado do prompt é passado como entrada para o model (o LLM). O modelo, por sua vez, gera uma ChatMessage.

* **Saída**: O resultado do model é a saída da cadeia.

### Quando usar o LCEL?

* **Construção de Cadeias Simples e Complexas**: Se você precisa conectar prompts, modelos, parsers, ferramentas ou outros componentes do LangChain em uma sequência, o LCEL é a melhor escolha. A sintaxe | é muito mais legível do que aninhar chamadas de funções.

* **Aplicações que Exigem Desempenho**: Graças ao paralelismo e ao suporte a streaming, o LCEL é ideal para aplicações em produção onde o tempo de resposta é crucial, como chatbots e assistentes virtuais.

* **Prototipagem Rápida**: Para testar rapidamente diferentes combinações de prompts e modelos, o LCEL permite montar e desmontar cadeias de forma ágil.

* **Código Limpo e Modular**: Se você valoriza a legibilidade e a manutenção do código, o LCEL força uma estrutura mais organizada, onde cada parte da cadeia é um componente claramente definido.

Vamos a um exemplo prático:

### LLM's que vamos utilizar durante todo o curso

Imports
```python
# Modelos LLM (Large Language Models)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
```

```python
# llms.py

# Cria os modelos do Curs
llm_padrao = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_API_KEY'), model_name='llama-3.3-70b-versatile')        # .env/Secret = GROQ_API_KEY
llm_openai = ChatOpenAI(temperature=0, model="gpt-4o-mini")                                                               # .env/Secret = OPENAI_API_KEY
llm_gemini = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-flash-latest')                                       # .env/Secret = GOOGLE_API_KEY
llm_groq_p = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_API_KEY'), model_name='deepseek-r1-distill-llama-70b')  # .env/Secret = GROQ_API_KEY
```

```python
# exemplo_02.py

set_debug(False)

# Criando o agente
def agente_lcel(pergunta:str) -> str:
    modelo:str = llm_padrao # Groq
    prompt:str = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é um assistente inteligente com acesso a duas ferramentas:"),
            ("system", "1. Calculadora"),
            ("system", "2. Wikipedia"),
            ("system", "Dado a pergunta abaixo, diga o que pretende fazer."),
            ("human", "{pergunta}"),
            ("system", "Responda no formato:"),
            ("system", "Ação: [Calculadora|Wikipedia|Responder diretamente]"),
            ("system", "Motivo: ..."),
        ])

    # LCEL (LangChain Expression Language)
    cadeia = prompt | modelo | StrOutputParser()
    return cadeia.invoke({"pergunta": pergunta})
```

```python
# Utilizando nosso primeiro agente
resposta = agente_manual("Qual a raiz quadrada de 256?")
print(resposta)
```

## AgentExecutor


O **AgentExecutor** é o motor de execução de um agente. Ele é a lógica de alto nível que orquestra o processo de tomada de decisão. As principais responsabilidades do AgentExecutor são:

* **Observar o Histórico de Conversas**: Ele recebe o prompt do usuário e o histórico da conversa.

* **Chamar o Agent**: Ele envia essa informação para o Agent (que é um Runnable, ou seja, pode ser construído com LCEL). O Agent é a "**mente**" que decide a próxima ação.

* **Processar a Resposta do Agente**: A resposta do Agent pode ser uma de duas coisas:
> * **Uma AgentAction**: O agente decidiu usar uma ferramenta. O AgentExecutor então chama a ferramenta especificada com a entrada correta.
> * **Uma AgentFinish**: O agente decidiu que a tarefa está completa e tem a resposta final para o usuário.

* **Loop**: Se for uma **AgentAction**, o `AgentExecutor` executa a ferramenta, obtém o resultado e repete o processo (volta para o **passo 1**), enviando o resultado da ferramenta de volta para o agente. Ele faz isso em um loop até que o agente decida que a tarefa está finalizada (**AgentFinish**).

Resumindo o `AgentExecutor` é a camada que gerencia o ciclo de vida do agente, o "ciclo de raciocínio".


### **Qual é melhor utilizar o `LCEL` ou o `AgentExecutor`?**

Você não precisa escolher entre LCEL e AgentExecutor. O AgentExecutor é, na verdade, uma implementação de uma cadeia construída com LCEL, porém com uma lógica de alto nível para gerenciar o ciclo de vida do agente.

#### Exemplo do AgentExecutor

```python
# exemplo_02.1.py

# Visualizar os detalhes da execução
set_debug(False)

def agente_langchain(pergunta:str) -> dict:
    modelo = llm_padrao # Groq
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é um assistente inteligente com acesso a duas ferramentas:"),
            ("system", "1. Calculadora"),
            ("system", "2. Wikipedia"),
            ("system", "Dado a pergunta abaixo, diga o que pretende fazer."),
            ("human", "{pergunta}"),
            ("system", "Responda no formato:"),
            ("system", "Ação: [Calculadora|Wikipedia|Responder diretamente]"),
            ("system", "Motivo: ... "),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Onde o agente irá escrever suas anotações (Pensamento)
        ]
    )
    agente = create_tool_calling_agent(modelo, tools=[], prompt=prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=[])
    resposta = executor_do_agente.invoke({"pergunta": pergunta})
    return resposta['output']
```

```python
resposta = agente_langchain("Qual a raiz quadrada de 256?")
print(resposta)
```


## Algumas vantagens do LangChain (Portabilidade)

LLM + Tool + Prompt

***IMPORTANTE: Este exemplo precisa de uma chave da OpenAI e Gemini.***

```python
# exemplo_03.py

# Visualizar os detalhes da execução
set_debug(False)

def agente_langchain(llm:BaseChatModel, pergunta:str) -> dict:
    # https://python.langchain.com/docs/integrations/tools/
    # https://python.langchain.com/docs/versions/migrating_chains/llm_math_chain/
    ferramentas = load_tools(["llm-math"], llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é um agente responsável por resolver problemas matemáticos."),
            ("system", "Utilize todas as suas ferramentas disponíveis e responda a pergunta do usuário."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Onde o agente irá escrever suas anotações (Pensamento)
        ]
    )
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas)
    resposta = executor_do_agente.invoke({"input": pergunta})
    return resposta
```

```python
# Utilizando a LLM da OpenAI para responder
resposta = agente_langchain(llm_openai, "Qual é a raiz quadrada de 169 vezes 2?")
print(f'\n\nResposta OpenAi: {resposta.get("output", "Não encontrei a resposta")}\n')
```

```python
# Utilizando a LLM do Gemini para responder
resposta = agente_langchain(llm_gemini, "Qual é a raiz quadrada de 169 vezes 2?")
print(f'\nResposta Gemini: {resposta.get("output", "Não encontrei a resposta")}\n')
```

## Conceitos Importantes

### LLM - Temperature

```python
llm_padrao = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_API_KEY'), model_name='llama3-70b-8192')
llm_openai = ChatOpenAI(temperature=0, model="gpt-4o-mini")                                   
llm_gemini = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-flash-latest')   
```

Vamos entender o que acontece quando alteramos a temperatura da LLM. Sabemos que a **LLM** prevê sempre a próxima palavra. Exemplo:

```sh
Pergunta: Como está o dia hoje?
LLM: Hoje <>
LLM: Hoje o <>
LLM: Hoje o dia <>
LLM: Hoje o dia está <>
LLM: Hoje o dia está lindo.
Resposta: Hoje o dia está lindo.
```


![temperatura](https://middleware.datah.ai/agent_figura_02.png?12)






![grafico](https://middleware.datah.ai/agent_figura_03.png?12)


### Top_p (Amonstragem de núcleo)

* Dada a lista de **palavras ordenadas** da **maior para menor** probabilidade.
* Seleciona um **subconjunto** onde a soma das probabilidades é **maior ou igual ao top_p**
* O modelo escolhe aleatoriamente uma dessas palavras.

\

#### Exemplo = Top_p = 95%

Hoje o dia está
* **Lindo** 50%
* **Bonito** 30%
* **Claro** 15%


---


* Escuro 4%
* Banana 1%



### Min_p (Controlar a diversidade)

* Recupera a **palavra com maior probabilidade**.
* Define um limite de corte $$ \ {p_{max} * min_p}$$
* Seleciona um **subconjunto** onde a probabilidade é **maior que o limite de corte**.
* O modelo escolhe aleatoriamente uma dessas palavras.

#### Exemplo = Min_p = 5%

Hoje o dia está
* **Lindo** 50%    (Limite de corte 2.5%)
* **Bonito** 30%
* **Claro** 15%
* **Escuro** 4%

---


* Banana 1%


## Ferramentas

No contexto do LangChain, as **ferramentas** (ou tools) são funções que um modelo de linguagem (LLM) pode chamar para interagir com o mundo exterior. Pense nelas como os "**sentidos**" e "**mãos**" do seu agente de IA.

Alguns exemplos de ferramentas comuns incluem:

* **Busca na internet**: Uma ferramenta que usa um buscador como o Google ou o DuckDuckGo para encontrar informações atualizadas.
* **Calculadora**: Uma ferramenta que executa operações matemáticas precisas.
* **API de clima**: Uma ferramenta que faz uma chamada a uma API para obter a previsão do tempo para uma cidade.
* **Leitor de arquivos**: Uma ferramenta que permite ao agente ler o conteúdo de um documento.
* **Ferramenta de SQL**: Uma ferramenta que executa consultas em um banco de dados.


### Criando nossa primeira ferramenta

```python
# exemplo_04.py

# Visualizar os detalhes da execução
set_debug(False)

@tool
def get_current_time(*args, **kwargs) -> str:
    """O objetivo dessa ferramenta é retornar a data e hora atual."""
    now = datetime.datetime.now()
    return f"A data e hora atual é {now.strftime('%Y-%m-%d %H:%M:%S')}"


def agente_langchain(llm:BaseChatModel, usar_ferramentas:bool=True) -> dict:
    ferramentas = [get_current_time] if usar_ferramentas else []
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Onde o agente irá escrever suas anotações (Pensamento)
        ]
    )
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas)
    return executor_do_agente
```

```python
executor_do_agente = agente_langchain(llm_padrao, usar_ferramentas=False)
resposta = executor_do_agente.invoke({"input": "Qual é a data inicial e final dessa semana?"})
print(f'\n\nResposta sem Ferramenta: {resposta.get("output", "Não encontrei a resposta")}\n')
```

```python
set_debug(False)

executor_do_agente = agente_langchain(llm_padrao, usar_ferramentas=True)
resposta = executor_do_agente.invoke({"input": "Qual é a data inicial e final dessa semana?"})
print(f'\n\nResposta com Ferramenta: {resposta.get("output", "Não encontrei a resposta")}\n')
```

```python
resposta = executor_do_agente.invoke({"input": "Qual foi minha ultima pergunta?"})
print(f'\n\nResposta sem Memória: {resposta.get("output", "Não encontrei a resposta")}\n')
```

## Memória

A **memória** (ou memory) no LangChain é o componente que permite que os agentes e as cadeias de conversa retenham informações de interações anteriores. Sem a memória, cada interação seria tratada como uma nova e isolada, fazendo com que o LLM "**esquecesse**" o que foi dito nos turnos anteriores.

A memória é crucial para construir chatbots e assistentes que podem ter conversas fluidas e contextuais. Ela injeta o histórico da conversa no prompt de cada nova chamada ao LLM, permitindo que o modelo use esse contexto para gerar respostas mais relevantes.

Existem vários tipos de memória no LangChain, cada um com uma estratégia diferente para armazenar e recuperar o histórico:

* **ConversationBufferMemory**: A forma mais simples de memória. Ela armazena todas as mensagens da conversa em uma variável e as injeta no prompt. É fácil de usar, mas pode se tornar ineficiente para conversas muito longas, pois o tamanho do prompt cresce.

* **ConversationBufferWindowMemory**: Similar à anterior, mas armazena apenas as últimas N interações (uma "janela" de conversa). Isso evita que o prompt fique grande demais, mantendo apenas o contexto mais recente.

* **ConversationSummaryMemory**: Em vez de armazenar a conversa inteira, ela cria um resumo contínuo das interações anteriores. Isso é ótimo para conversas longas, pois mantém o contexto sem sobrecarregar o prompt.

* **ConversationSummaryBufferMemory**: Uma combinação das duas últimas, que armazena as interações recentes na íntegra e resume as interações mais antigas.



## Contexto

O termo "**contexto**" no mundo de modelos de linguagem e inteligência artificial refere-se a toda a informação relevante que um modelo precisa para entender e gerar uma resposta adequada para uma determinada solicitação. É o conjunto de dados que fornece o pano de fundo para a interação atual.

Em LangChain, o **contexto é tudo aquilo que você alimenta o modelo de linguagem (LLM) junto com a pergunta** do usuário para que ele possa dar uma resposta precisa. Ele pode vir de diversas fontes e é fundamental para que o LLM não responda com base apenas em seu conhecimento pré-treinado, mas sim com base nas informações que você forneceu.


![grafico](https://middleware.datah.ai/agent_figura_04.png?12)

```python
# exemplo_05.py

# Visualizar os detalhes da execução
set_debug(False)

@tool
def get_current_time(*args, **kwargs) -> str:
    """O objetivo dessa ferramenta é retornar a data e hora atual."""
    now = datetime.datetime.now()
    return f"A data e hora atual é: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def agente_langchain(llm:BaseChatModel, usar_ferramentas:bool=True) -> dict:
    ferramentas = [get_current_time] if usar_ferramentas else []

    # Retorna o histórico como uma lista de objetos de mensagem
    memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"), # O placeholder para o histórico
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Onde o agente irá escrever suas anotações (Pensamento)
        ]
    )
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas, memory=memoria)
    return executor_do_agente
```

```python
executor_do_agente = agente_langchain(llm_padrao, usar_ferramentas=True)
```

```python
resposta = executor_do_agente.invoke({"input": "Qual é a data inicial e final dessa semana?"})
print(f'\n\nResposta: {resposta.get("output", "Não encontrei a resposta")}\n')
```

```python
resposta = executor_do_agente.invoke({"input": "Qual foi minha ultima pergunta?"})
print(f'\n\nResposta com Memória: {resposta.get("output", "Não encontrei a resposta")}\n')
```

## Prompt

O "**prompt**" é a instrução, pergunta ou texto inicial que você fornece a um modelo de linguagem (LLM) para que ele gere uma resposta. É o ponto de partida de qualquer interação com uma IA generativa.

Ele é o principal meio de comunicação com a LLM, e a qualidade da sua resposta depende, em grande parte, da clareza e da precisão do prompt. Um prompt bem elaborado pode guiar o modelo a entregar exatamente o que você precisa, enquanto um prompt vago pode levar a uma resposta genérica ou irrelevante.

### Tipos de Prompts

Os prompts podem ser categorizados de diferentes formas, dependendo do seu formato e da informação que contêm. No contexto do LangChain e do desenvolvimento com LLMs, as duas categorias mais importantes são:

* **Prompts Simples (Strings)**: Este é o tipo mais básico de prompt. É uma string de texto simples que você envia diretamente para o modelo. Não há formatação complexa ou variáveis.
* **Prompts Estruturados (Templates)**: Este tipo de prompt é uma estrutura reutilizável, ou um template, que contém espaços reservados para variáveis. Em vez de escrever o prompt completo a cada vez, você preenche essas variáveis com dados dinâmicos. Essa abordagem é a mais utilizada em aplicações reais, pois permite criar prompts robustos e flexíveis.


#### Prompt Simples (String)

```python
# exemplo_06.py

# Criando o agente
def agente(pergunta:str) -> str:

    # [HumanMessage(content=pergunta)] == [("human", f"{pergunta}")]
    resposta = llm_padrao.invoke([HumanMessage(content=pergunta)])
    return resposta
```

```python
resposta = agente("Qual a raiz quadrada de 256?")
print(resposta.content)
```

#### Prompt Template

```python
# exemplo_07.py


# Criando o agente
def agente(pergunta:str) -> str:
    prompt = PromptTemplate(
        input_variables=["pergunta", "agent_scratchpad"],
        template="""Você é um assistente inteligente com acesso a duas ferramentas:
                        1. Calculadora
                        2. Wikipedia
                    Dado a pergunta abaixo, diga o que pretende fazer.
                    Pergunta: {pergunta}
                    {agent_scratchpad}
                    Responda no formato:
                    Ação: [Calculadora|Wikipedia|Responder diretamente]
                    Motivo: ...
        """
    )
    agente = create_tool_calling_agent(llm_padrao, tools=[], prompt=prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=[])
    resposta = executor_do_agente.invoke({"pergunta": pergunta})
    return resposta
```

```python
resposta = agente("Qual a raiz quadrada de 256?")
print(resposta.get("output", "Não encontrei a resposta"))
```

#### ChatPromptTemplate

```python
# exemplo_08.py

# Criando o agente
def agente(pergunta:str) -> str:
    prompt:str = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("Você é um assistente inteligente com acesso a duas ferramentas:"),
            SystemMessagePromptTemplate.from_template("1. Calculadora"),
            SystemMessagePromptTemplate.from_template("2. Wikipedia"),
            SystemMessagePromptTemplate.from_template("Dado a pergunta abaixo, diga o que pretende fazer."),
            HumanMessagePromptTemplate.from_template("{pergunta}"),
            SystemMessagePromptTemplate.from_template("Responda no formato:"),
            SystemMessagePromptTemplate.from_template("Ação: [Calculadora|Wikipedia|Responder diretamente]"),
            SystemMessagePromptTemplate.from_template("Motivo: ..."),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    agente = create_tool_calling_agent(llm_padrao, [], prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=[])
    resposta = executor_do_agente.invoke({"pergunta": pergunta})
    return resposta
```

```python
resposta = agente("Qual a raiz quadrada de 256?")
print(resposta.get("output", "Não encontrei a resposta"))
```

## RAG (Retrieval-Augmented Generation)

**RAG**, ou **Geração Aumentada por Recuperação**, é uma técnica que combina o poder de um modelo de linguagem (LLM) com sistemas de recuperação de informações. Em termos simples, o RAG permite que o LLM acesse dados externos, como seus próprios documentos, bases de conhecimento ou a internet, antes de gerar uma resposta.

O RAG **resolve três grandes problemas** dos LLMs tradicionais:

1. **Conhecimento Desatualizado**: LLMs são treinados em grandes volumes de dados, mas esse conhecimento é estático e limitado à data do treinamento. O RAG permite que o modelo acesse informações em tempo real e dados que são constantemente atualizados.

2. **Alucinações**: Como os LLMs às vezes inventam informações para preencher lacunas, eles podem gerar respostas incorretas ou sem fundamento. O RAG "aterra" a resposta em fatos concretos, usando as informações recuperadas de uma fonte externa confiável, o que reduz drasticamente a chance de alucinações.

3. **Falta de Transparência**: Com o RAG, o modelo não apenas responde, mas também pode citar as fontes de onde a informação foi extraída. Isso aumenta a confiança do usuário, pois ele pode verificar a veracidade da resposta.

Como o RAG funciona?

![rag](https://media.geeksforgeeks.org/wp-content/uploads/20250210190608027719/How-Rag-works.webp)

Mais informações https://www.geeksforgeeks.org/nlp/what-is-retrieval-augmented-generation-rag/

### Criando o Banco de dados vetorial (Chroma DB)

```python
# exemplo_09.py

# Visualizar os detalhes da execução
set_debug(False)


def cria_banco_de_dados_vetorial(path_documentos:str) -> None:
    try:
        # Carrega os documentos do diretório especificado
        documents = PyPDFDirectoryLoader(path_documentos).load()

        # Usando embeddings do OpenAI
        embeddings = OpenAIEmbeddings()

        # Cria um banco de dados vetorial usando Chroma
        split_documents = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(documents)

        # Cria o banco de dados vetorial
        vectorstore = Chroma.from_documents(split_documents, embeddings, persist_directory=f'{OUTPUT_DOCUMENTS_DIR}vectorstore')

        print("Banco de dados vetorial criado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar documentos: {e}")
```

```python
cria_banco_de_dados_vetorial(path_documentos=OUTPUT_DOCUMENTS_DIR)
```

### Carregando o banco vetorial criado

```python
# exemplo_10.py

# Visualizar os detalhes da execução
set_debug(False)


def carrega_banco_de_dados_vetorial(path_documentos:str) -> Chroma:
    try:
        # Carrega o banco de dados vetorial existente
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=path_documentos, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Erro ao carregar o banco de dados vetorial: {e}")
        return None
```

```python
vectorstore = carrega_banco_de_dados_vetorial(f'{OUTPUT_DOCUMENTS_DIR}vectorstore')
docs = None

if vectorstore:
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke("Data H")
    print(docs)
else:
    print("Não foi possível carregar o banco de dados vetorial.")
```

### Criando o Agente com o RAG

![rag](https://media.geeksforgeeks.org/wp-content/uploads/20250210190608027719/How-Rag-works.webp)

#### Carregando as bibliotecas

```python
# exemplo_11.py


# Visualizar os detalhes da execução
set_debug(False)
```

#### Carregando o banco vetorial

```python
def carrega_banco_de_dados_vetorial(path_documentos:str) -> Chroma:
    try:
        # Carrega o banco de dados vetorial existente
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=path_documentos, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Erro ao carregar o banco de dados vetorial: {e}")
        return None

```

### Busca os dados e Cria o Contexto

```python
def busca_na_base_de_documentos(pergunta:str) -> str:
    """Use esta ferramenta para responder perguntas sobre a Data H, seus produtos como NIC, Consultoria, Cyber Segurança,
       ou qualquer informação contida na base de conhecimento. A entrada deve ser a pergunta do usuário."""
    vectorstore = carrega_banco_de_dados_vetorial(f'{OUTPUT_DOCUMENTS_DIR}vectorstore')
    contexto = None
    if vectorstore:
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(pergunta)
        contexto = "\n\n".join([doc.page_content for doc in docs])
    return contexto

```

#### Cria o agente

```python
def agente_langchain(llm:BaseChatModel) -> dict:
    ferramentas = []
    memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input") # Retorna o histórico como uma lista de objetos de mensagem
    prompt = PromptTemplate(
        input_variables=["input", "context", "chat_history", "agent_scratchpad"], # Variáveis de entrada
        template="""{chat_history}
            Você é um agente de IA especializado em responder perguntas.
            Contexto: {context}
            Pergunta: {input}
            {agent_scratchpad}
        """
    )
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas, memory=memoria)
    return executor_do_agente
```

### Testando o RAG

```python
executor_do_agente = agente_langchain(llm_padrao)

pergunta = "O que é o NIC?"
```

#### Sem contexto

```python
contexto = ''
resposta = executor_do_agente.invoke({"input": pergunta, "context": contexto})
print(f'\n\nResposta sem Contexto: {resposta.get("output", "Não encontrei a resposta")}\n')
```

#### Com contexto RAG

```python
contexto = busca_na_base_de_documentos(pergunta) or ''
resposta = executor_do_agente.invoke({"input": pergunta, "context": contexto})
print(f'\n\nResposta com Contexto: {resposta.get("output", "Não encontrei a resposta")}\n')
```

```python
pergunta = "De qual empresa é esse produto e onde ela fica?"
contexto = busca_na_base_de_documentos(pergunta) or ''
resposta = executor_do_agente.invoke({"input": pergunta, "context": contexto})
print(f'\n\nResposta com Contexto e Memória: {resposta.get("output", "Não encontrei a resposta")}\n')
```

## Tipos de Agentes

### AgentExecutor

* `AgentExecutor` (orquestrador): é a classe principal no LangChain responsável por orquestrar todo o ciclo de vida de um agente. Ele é o "**motor**" que gerencia o fluxo de trabalho. Sua função é:

\

![grafico](https://middleware.datah.ai/agent_figura_05.png?12)

### ReAct

* **ReAct** (O Padrão de Raciocínio) é um acrônimo para Reasoning and Acting (Raciocínio e Ação). Ele é um padrão de pensamento que um agente segue para tomar decisões. No padrão ReAct, o agente não apenas responde, ele “**pensa em voz alta**”:

\

![grafico](https://middleware.datah.ai/agent_figura_06.png?12)

\


* **Pensamento**: O agente descreve o seu raciocínio. Ele analisa a pergunta e decide qual seria o próximo passo.
* **Ação**: O agente decide qual ferramenta usar e com quais argumentos.
* **Observação**: A saída da ferramenta. É o resultado real da ação.
* **Resposta Final**: Quando o agente determina que a tarefa está concluída, ele para de raciocinar e fornece a resposta ao usuário.





### Outros Tipos

O **ReAct** é o padrão de raciocínio mais popular, mas o `AgentExecutor` no LangChain pode ser configurado com **outros tipos de lógica de agente**. Os mais comuns são:

* **ReAct Zero-shot**: A versão mais básica do ReAct, onde o agente decide a ação com base apenas na sua capacidade de raciocínio.
* **Conversational ReAct**: Uma extensão do ReAct que usa memória e histórico de conversa, tornando-o ideal para chatbots.
* **OpenAI Functions / OpenAI Tools**: Este é um tipo de agente que se baseia na funcionalidade de "**chamada de função**" dos modelos da OpenAI. Em vez de o agente gerar um texto no formato **Thought/Action**, o próprio modelo gera uma chamada de função estruturada (**uma AgentAction**) que o `AgentExecutor` então executa. É uma abordagem mais direta.

```python
# exemplo_12.py

# Visualizar os detalhes da execução
set_debug(True)


def agente_langchain(llm:BaseChatModel) -> dict:
    ferramentas = [PythonAstREPLTool()]

    prompt = hub.pull("hwchase17/react")
    print('\n','-'*40,'\n',prompt.template, '\n','-'*40, '\n')

    agente = create_react_agent(llm, ferramentas, prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas, handle_parsing_errors=True)
    return executor_do_agente

```

```python
executor_do_agente = agente_langchain(llm_groq_p)  # DeepSeek R1

pergunta = "Qual é a área do triângulo com base 10 e altura 5?"

resposta = executor_do_agente.invoke({"input": pergunta})
print(f'\n\nResposta DeepSeek R1: {resposta.get("output", "Não encontrei a resposta")}\n')
```

## MCP - Model Context Protocol

O MCP (**Model Context Protocol**) é um protocolo aberto que visa padronizar a forma como aplicações de IA, como agentes, interagem com ferramentas externas e fontes de dados.

Pense no MCP como um "**adaptador universal**" para a IA. Em vez de cada aplicação de IA ter que ser codificada para se comunicar com centenas de APIs de ferramentas diferentes, o MCP oferece uma interface comum. Isso permite que qualquer modelo de linguagem que entenda o protocolo possa usar qualquer ferramenta compatível com o MCP, independentemente de quem as criou.


### Integração sem MCP


\

![mcp](https://middleware.datah.ai/agent_figura_07.png?12)

### Integração com MCP

\

![mcp2](https://middleware.datah.ai/agent_figura_08.png?12)

### Protocolos MCP

* **SSE (Server-Sent Events)**
É um protocolo de comunicação que funciona sobre HTTP. Sua principal característica é a comunicação unidirecional, onde o servidor envia dados para o cliente em tempo real, através de uma conexão HTTP persistente.

* **STDIO (Standard Input/Output)**
É um conceito de comunicação fundamental em sistemas operacionais, não um protocolo de rede. Ele se refere aos canais de comunicação padrão de um programa: **stdin** (entrada padrão), **stdout** (saída padrão) e **stderr** (saída de erro padrão).

### Diferenças entre os protocolos

| Característica | SSE                                    | STDIO                               |
|----------------|----------------------------------------|-------------------------------------|
| Ambiente       | Comunicação de rede (cliente-servidor) | Comunicação local (inter-processos) |
| Fluxo          | Unidirecional (servidor -> cliente)    | Bidirecional (leitura e escrita)    |
| Protocolo      | HTTP                                   | Canais de sistema operacional       |
| Uso em IA      | Streaming de respostas de LLMs         | Comunicação com ferramentas locais  |

### Exemplo de utilização no Cursor AI.

```python
{
  "mcpServers": {
      "local-server-tools": {
        "command": "c:/Dados/Cursos/ia/.venv/Scripts/python.exe",
        "args": ["c:/Dados/Cursos/ia/mcp/local_server.py"]
      },
      "datah": {
            "url": "http://127.0.0.1:5008/sse",
            "transport": "sse"
      }
  }
}

```

### Vamos para o código

***IMPORTANTE: Esse código deve ser rodado localmente e não no colab.***

#### Rotinas de apoio.

Não deixa na ferramenta toda a responsabilidade, transforma em componentes para serem reutilizados.

```python
# mcp_helpers.py

class ArxivHelper:

    @property
    def base_url(self):
        return 'https://export.arxiv.org/api/query'

    async def make_arxiv_request(self, url:str) -> str | None:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers={"User-Agent": 'arxiv-search-app/1.0'}, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(e)
                return None

    def parse_arxiv_response(self, xml_data:str) -> list[dict[str, Any]]:
        if not xml_data:
            return []

        root = ET.fromstring(xml_data)

        namespaces = {
            "atom": 'http://www.w3.org/2005/Atom',
            "arxiv": 'http://arxiv.org/schemas/atom'
        }

        entries = []
        for entry in root.findall('.//atom:entry', namespaces):
            e_title = entry.find('atom:title', namespaces)
            e_summary = entry.find('atom:summary', namespaces)
            e_link =  entry.find('atom:id', namespaces)
            e_published = entry.find('atom:published', namespaces)

            title = e_title.text.strip() if e_title is not None else ""
            summary = e_summary.text.strip() if e_summary is not None else ""

            authors = []
            for author in entry.findall('.//atom:author/atom:name', namespaces):
                authors.append(author.text.strip())

            link = e_link.text.strip() if e_link is not None else ""
            published = e_published.text.strip() if e_published is not None else ""

            entries.append(dict(
                title=title,
                summary=summary,
                authors=authors,
                link=link,
                published=published
            ))

        return entries

    def format_paper(self, paper:dict) -> str:
        authors_str = " ".join(paper.get('authors', ['Unknown author']))
        return f"""
            title: {paper.get('title', '')}
            authors: {authors_str}
            published: {paper.get('published', '')[:10]}
            link: {paper.get('link', '')}
            summary: {paper.get('summary', '')}
        """


class MailRecipient(object):

    def __init__(self):
        self.__recipients: List = []

    def add(self, email: str, name: str = None):
        if not name:
            self.__recipients.append(email)
        else:
            self.__recipients.append(formataddr((name, email)))

    def clear(self):
        self.__recipients = []

    def get(self) -> str:
        return ', '.join(self.__recipients)

    def has_item(self) -> bool:
        if not self.__recipients:
            return False
        return len(self.__recipients) > 0


class MailMessage(object):

    def __init__(self, sender_email: str, sender_name: str = None):
        self.__message: MIMEMultipart = MIMEMultipart()
        self.__from: str = sender_email
        self.__from_name: str = sender_name
        self.__body: Union[str, None] = None
        self.__subject: Union[str, None] = None
        self.to: MailRecipient = MailRecipient()
        self.cc: MailRecipient = MailRecipient()
        self.bcc: MailRecipient = MailRecipient()

    def get_message(self) -> MIMEMultipart:
        self.__message['From'] = formataddr((self.__from_name, self.__from)) if self.__from_name else self.__from
        self.__message['Subject'] = self.__subject
        self.__message['To'] = self.to.get()
        if self.cc.has_item():
            self.__message['Cc'] = self.cc.get()
        if self.bcc.has_item():
            self.__message['Bcc'] = self.bcc.get()
        self.__message.attach(self.__body)
        self.__validate_mail_message()
        return self.__message

    def set_subject(self, subject: str):
        self.__subject = subject

    def set_text_body(self, text):
        self.__body = MIMEText(text, "plain")

    def set_html_body(self, html):
        self.__body = MIMEText(html, "html")

    def attach_file(self, filename: str, mime_type: str = "application/octet-stream"):
        with open(filename, 'rb') as attachment:
            mime_type_parts: List[str] = mime_type.split('/')
            part: MIMEBase = MIMEBase(mime_type_parts[0], mime_type_parts[1])
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {Path(filename).name}")
        self.__message.attach(part)

    def __validate_mail_message(self):
        if not self.__subject:
            raise ValueError("The email subject is required.")
        if not self.__body:
            raise ValueError("The email body is required.")
        if not self.__message['From']:
            raise ValueError('From is required.')
        if len(self.__message['From']) == 0:
            raise ValueError("The sender (from) is required.")
        if not self.to.has_item() and not self.cc.has_item() and not self.bcc.has_item():
            raise ValueError("Add at least a email to send this message.")


class SMTPServer(object):

    def __init__(self, host: str, port: int = 587,
                 username: str = None, password: str = None,
                 has_ssl: bool = False, has_tls: bool = True,
                 has_authentication: bool = True):

        self.__host: str = host
        self.__port: int = port
        self.__username: str = username
        self.__password: str = password
        self.__tls: bool = has_tls
        self.__ssl: bool = has_ssl
        self.__authentication: bool = has_authentication
        self.__context = ssl.create_default_context()

    def connect(self) -> Union[SMTP, SMTP_SSL]:
        try:
            if self.__tls:
                self.__server = SMTP(host=self.__host, port=self.__port)
                self.__server.ehlo()
                self.__server.starttls(context=self.__context)
                self.__server.ehlo()
            elif self.__ssl:
                self.__server = SMTP_SSL(host=self.__host, port=self.__port, context=self.__context)
            else:
                self.__server = SMTP(host=self.__host, port=self.__port)

            if self.__authentication:
                self.__server.login(user=self.__username, password=self.__password)

        except Exception as e:
            if self.__server:
                self.disconnect()
            raise e
        return self.__server

    def disconnect(self):
        self.__check_connection()
        self.__server.quit()

    def send(self, mail_message: MailMessage) -> bool:
        self.__check_connection()
        self.__server.send_message(msg=mail_message.get_message())
        return True

    def get_sender(self) -> str:
        return self.__username

    def __check_connection(self):
        if not self.__server:
            raise ConnectionError("The server is not connected. Connect first.")


```

#### Criando o Servidor MCP

***IMPORTANTE: Não dá para rodar no colab, apenas local.***

```python
# mcp_server.py

async def search_arxiv_tool(query:str, max_results:int = 5) -> str:
    """
    Esta ferramenta busca por artigos científicos no site arxiv.org

    Args:
        query (str): O assunto que deseja buscar no site.
        max_results (int, optional): Quantidade máxima de artigos retornados pelo site. O padrão é 5.
    """
    try:
        arxiv_mcp_tool = ArxivHelper()
        formatted_query = query.replace(" ", '+')

        url = f"{arxiv_mcp_tool.base_url}?search_query=all:{formatted_query}&start=0&max_results={max_results}"
        xml_data = await arxiv_mcp_tool.make_arxiv_request(url)
        if not xml_data:
            raise ValueError("Não foi capaz de recuperar os dados do arxiv.")

        papers = arxiv_mcp_tool.parse_arxiv_response(xml_data)
        if not papers:
            return FileNotFoundError("Artigos não encontrados.")

        paper_texts = [arxiv_mcp_tool.format_paper(paper) for paper in papers]
        return "\n---\n".join(paper_texts)
    except Exception as e:
        return f"ERRO: {str(e)}"


async def send_mail(subject:str, email_to:str, email_content:str, email_attach_file:str=None) -> str:
    """
    Esta ferramenta envia um e-mail para uma pessoa.

    Args:
        subject (str): É o assunto do email, e deve ser um título curto.
        email_to (str): É o e-mail para quem será enviado o email. Este argumento pode receber mais de um email, em uma string separados por "," or ";". Ex: usuario1@domain.com, usuario2@domain.com
        email_content (str): É o conteúdo do email
        email_attach_file (str, optional): É o caminho absoluto de um arquivo para anexar ao email. Se não existir anexo, o valor deve ser None. O valor padrão é None.
    """
    username=os.getenv('SMTP_USERNAME')
    password=os.getenv('SMTP_PASSWORD')
    smtp = SMTPServer(host='smtp.gmail.com', port=587, username=username, password=password, has_ssl=True, has_tls=True, has_authentication=True)

    sender_email = 'marcelopiovan@gmail.com'
    sender_name = 'Marcelo Piovan'

    message = MailMessage(sender_email=sender_email, sender_name=sender_name)
    message.set_subject(subject=subject)
    for email in re.split(r'[,;]', email_to):
        email = email.strip()
        if email:
            message.to.add(email=email)

    message.set_html_body(email_content)

    if email_attach_file is not None and email_attach_file != '' and len(email_attach_file) > 0:
        if not os.path.isfile(email_attach_file):
            raise FileNotFoundError(f"Arquivo não encontrado: {email_attach_file}")
        message.attach_file(filename=email_attach_file)

    try:
        smtp.connect()
        smtp.send(message)
        smtp.disconnect()
        return 'Email enviado com sucesso!'
    except Exception as e:
        return f"ERRO: {str(e)}"

```

#### Rodando o servidor

```python
if RUNNING_IN_COLAB:
    raise NotImplementedError('Esse servidor deve rodar localmente.')


if __name__ == "__main__":
    print("🚀 Iniciando o servidor ... ")
    mcp = FastMCP(name="DataH_MCP", port=5008)
    print(f'URL para verificação "http://127.0.0.1:5008/sse"')
    mcp.add_tool(search_arxiv_tool)
    mcp.add_tool(send_mail)
    mcp.run(transport='sse')
```

### Usando um Servidor MCP - Cliente

***IMPORTANTE: Não dá para rodar no colab, apenas local.***

```python
# mcp_client.py

set_debug(False)

server_params = {
    # "local-server-tools": {
    #     "command": "C:\\Dados\\Projetos\\aulas\\agent\\.venv\\Scripts\\python.exe",
    #     "args": ["C:/Dados/Projetos/aulas/agent/src/mcp_server.py"],
    #     "transport": "stdio",
    # },
    "server-tools": {
        "url": "http://127.0.0.1:5008/sse",
        "transport": "sse",
    }
}

async def run_agent():
    model:str = "gpt-4o-mini"
    checkpointer = InMemorySaver()

    async with MultiServerMCPClient(server_params) as client:

        all_tools = client.get_tools()
        if not all_tools:
            print("\033[31mNenhuma ferramenta disponível no servidor MCP.\033[0m")

        for server, tools in client.server_name_to_tools.items():
            print(f'\033[31m\n==== MCP Server UP! - {server} ====\033[0m')
            for tool in tools:
                print(f'\033[35m* {tool.name} *\033[0m\n{tool.description}\n')


        prompt = f"""
            Sua tarefa é solucionar as perguntas do usuário, usando as ferramentas disponíveis e seu próprio conhecimento.
            Responda sempre em português.
        """
        agent = create_react_agent_graph(model, all_tools, checkpointer=checkpointer, prompt=prompt)

        session_id = str(uuid4())
        config = {"configurable": {"thread_id": session_id}}

        while True:
            user_input = input("\033[33mFaça a sua pergunta: \033[0m")

            if user_input == "sair":
                break

            if user_input == "limpar":
                print("\033c")
                continue

            print(f"\033[34mUsuário: {user_input}\033[0m")

            agent_response = await agent.ainvoke({"messages": user_input}, config=config)
            print(f"\033[32mAgente: {agent_response['messages'][-1].content}\033[0m")

            checkpoint = await checkpointer.aget(config)


```

#### Rodando o Cliente

```python
if RUNNING_IN_COLAB:
    raise NotImplementedError('Esse cliente deve rodar localmente.')


if __name__ == "__main__":
    result = asyncio.run(run_agent())
```

## LangGraph

O **LangGraph** é uma biblioteca construída sobre o LangChain que serve para criar agentes e **fluxos de trabalho multi-etapa** de forma mais robusta e controlada. Em vez de modelar um agente como uma simples cadeia linear, o LangGraph o representa como um grafo de estados, permitindo criar lógicas complexas com nós e arestas.

\

![graph](https://middleware.datah.ai/graph.png?12)



Pense no `AgentExecutor` como um motor de carro que só sabe seguir um caminho em **linha reta** (o loop de raciocínio ReAct).

O `LangGraph` é como um **sistema de navegação completo** que permite ao motorista escolher caminhos diferentes, fazer retornos, parar em pontos de interesse e até mesmo mudar de plano no meio da jornada.

O `LangGraph` é ideal para construir agentes que precisam de lógica de controle complexa. Ele é a ferramenta para situações onde um simples AgentExecutor se torna limitado, como:

* **Lógica Condicional**: Criar agentes que podem tomar decisões de "se/então" em cada passo.

>> *Ex: Se a busca na ferramenta A falhar, tente a ferramenta B.*

* **Múltiplos Fluxos de Trabalho**: Modelar um agente que pode executar diferentes tarefas com base no input inicial.

>> *Ex: Se a pergunta for sobre finanças, siga um fluxo. Se for sobre atendimento ao cliente, siga outro.*

* **Ciclos de Conversa Complexos**: Gerenciar conversas que precisam de aprovação do usuário, feedback ou validação antes de continuar para o próximo passo.

* **Agentes de Longo Prazo**: Modelar sistemas que precisam manter um estado persistente e complexo ao longo de várias interações, como agentes que acompanham o progresso de um projeto.




O `LangGraph` usa o conceito de **nós** (as funções ou ações a serem executadas) e **arestas** (as transições entre os nós), com a habilidade de definir **condicional_edges** para ramificações dinâmicas no fluxo.

\

![grafico](https://middleware.datah.ai/agent_figura_09.png?12)

| Vantagens                                                                                                                                                                             | Desvantagens                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Controle Total do Fluxo: Você define cada etapa e transição do agente, eliminando o "efeito caixa preta" do AgentExecutor.                                                            | Complexidade Inicial: O aprendizado e a configuração inicial são mais complexos.                                                                                 |
| Robustez: É mais fácil de criar mecanismos de recuperação e tratamento de erros, direcionando o fluxo para um nó de tratamento de erros em caso de falha.                             | Não é para Todos os Casos: Para agentes ReAct simples ou cadeias lineares, o AgentExecutor com LCEL é mais do que suficiente e muito mais rápido de implementar. |
| Estado Gerenciado: Ele gerencia o estado completo do agente (incluindo o histórico de conversa e saídas de ferramentas) em um único objeto de State, tornando a depuração mais fácil. | Curva de Aprendizagem: Requer uma compreensão de conceitos de grafos e máquinas de estado, o que pode ser um obstáculo para iniciantes.                          |

```python
# exemplo_13.py

# Visualizar os detalhes da execução
set_debug(False)


# Formatação das respostas
def formatar_classificacao_para_estado(classificacao: str):
    palavra_chave = classificacao.lower().strip().split()[0]
    return {"classificacao": palavra_chave}

def formatar_resposta_caes(resposta: str):
    return {"resposta": resposta}

def formatar_resposta_gatos(resposta: str):
    return {"resposta": resposta}



# Cadeias para cada especialidade
def cadeia_cachorro(llm):
    prompt_caes = ChatPromptTemplate.from_template(
        "Você é um especialista em cães. Responda a pergunta a seguir de forma concisa: {pergunta}"
    )
    return prompt_caes | llm | StrOutputParser() | RunnableLambda(formatar_resposta_caes)


def cadeia_gato(llm):
    prompt_gatos = ChatPromptTemplate.from_template(
        "Você é um especialista em gatos. Responda a pergunta a seguir de forma concisa: {pergunta}"
    )
    return prompt_gatos | llm | StrOutputParser() | RunnableLambda(formatar_resposta_gatos)


def cadeia_classificador(llm):
    classificador_prompt = PromptTemplate.from_template(
        """Classifique a seguinte pergunta como 'caes' ou 'gatos'.
            Responda apenas com a palavra 'caes' ou 'gatos'.
        Pergunta: {pergunta}

        Tópico:"""
    )
    return classificador_prompt | llm | StrOutputParser() | RunnableLambda(formatar_classificacao_para_estado)


# Define o estado do nosso grafo
class GrafoState(TypedDict):
    pergunta: str
    classificacao: str
    resposta: str


# Condicional para rotear a pergunta
def rotear_pergunta(state):
    if "caes" in state["classificacao"].lower():
        return "cadeia_caes"
    elif "gatos" in state["classificacao"].lower():
        return "cadeia_gatos"
    else:
        # Padrão para cães se não conseguir classificar
        return "cadeia_caes"


def fluxo(llm):
    workflow = StateGraph(GrafoState)

    # Adiciona os nós (etapas)
    workflow.add_node("classificador", cadeia_classificador(llm))
    workflow.add_node("cadeia_caes", cadeia_cachorro(llm))
    workflow.add_node("cadeia_gatos", cadeia_gato(llm))

    # O início do grafo
    workflow.set_entry_point("classificador")

    workflow.add_conditional_edges(
        "classificador",
        rotear_pergunta,
        {
            "cadeia_caes": "cadeia_caes",
            "cadeia_gatos": "cadeia_gatos",
        },
    )

    # E os pontos de saída
    workflow.add_edge("cadeia_caes", END)
    workflow.add_edge("cadeia_gatos", END)

    # Compila o grafo para uso
    return workflow.compile()

```

```python
workflow = fluxo(llm_padrao)

# Mostra o fluxo
workflow.get_graph().draw_png("graph.png")
workflow.get_graph().print_ascii()

img = mpimg.imread('graph.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.title('Graph')
plt.show()
```

```python
# Executando o grafo com uma pergunta sobre cães
pergunta_caes = "Por que os cachorros gostam tanto de brincar de buscar?"
resultado_caes = workflow.invoke({"pergunta": pergunta_caes})
print(f"Pergunta: {pergunta_caes}")
print(f"Resposta: {resultado_caes['resposta']}\n")
```

```python
# Executando o grafo com uma pergunta sobre gatos
pergunta_gatos = "Qual é o som mais comum que os gatos fazem?"
resultado_gatos = workflow.invoke({"pergunta": pergunta_gatos})
print(f"Pergunta: {pergunta_gatos}")
print(f"Resposta: {resultado_gatos['resposta']}")
```

![grafico](https://middleware.datah.ai/agent_figura_10.png?12)

# Anexo I - Groq

## Criando uma conta no Groq para conseguirmos uma **Free API Key** 😎

![groq](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMiIgZGF0YS1uYW1lPSJMYXllciAyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMDAuMTggNjkuNzYiPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuY2xzLTEgewogICAgICAgIGZpbGw6ICNmZmY7CiAgICAgIH0KICAgIDwvc3R5bGU+CiAgPC9kZWZzPgogIDxnIGlkPSJMYXllcl8xLTIiIGRhdGEtbmFtZT0iTGF5ZXIgMSI+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xMTQuMjYuMTNjLTEzLjE5LDAtMjMuODgsMTAuNjgtMjMuODgsMjMuODhzMTAuNjgsMjMuOSwyMy44OCwyMy45LDIzLjg4LTEwLjY4LDIzLjg4LTIzLjg4aDBjLS4wMi0xMy4xOS0xMC43MS0yMy44OC0yMy44OC0yMy45Wk0xMTQuMjYsMzguOTRjLTguMjQsMC0xNC45My02LjY5LTE0LjkzLTE0LjkzczYuNjktMTQuOTMsMTQuOTMtMTQuOTMsMTQuOTMsNi42OSwxNC45MywxNC45M2MtLjAyLDguMjQtNi43MSwxNC45My0xNC45MywxNC45M2gwWiIvPgogICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjQuMTEsMEMxMC45Mi0uMTEuMTMsMTAuNDcsMCwyMy42NmMtLjEzLDEzLjE5LDEwLjQ3LDIzLjk4LDIzLjY2LDI0LjExaDguMzF2LTguOTRoLTcuODZjLTguMjQuMTEtMTUtNi41LTE1LjEtMTQuNzQtLjExLTguMjQsNi41LTE1LDE0Ljc0LTE1LjFoLjM0YzguMjIsMCwxNC45NSw2LjY5LDE0Ljk1LDE0LjkzaDB2MjEuOThoMGMwLDguMTgtNi42NSwxNC44My0xNC44MSwxNC45My0zLjkxLS4wNC03LjYzLTEuNTktMTAuMzktNC4zOGwtNi4zMyw2LjMxYzQuNCw0LjQyLDEwLjM0LDYuOTIsMTYuNTcsNi45OWguMzJjMTMuMDItLjE5LDIzLjQ5LTEwLjc1LDIzLjU2LTIzLjc3di0yMi42OUM0Ny42NSwxMC4zNSwzNy4wNS4wMiwyNC4xMSwwWiIvPgogICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMTkxLjI4LDY4Ljc0VjIzLjQzYy0uMzItMTIuOTYtMTAuOTItMjMuMjgtMjMuODgtMjMuMy0xMy4xOS0uMTMtMjMuOTgsMTAuNDctMjQuMTEsMjMuNjYtLjEzLDEzLjE5LDEwLjQ5LDIzLjk4LDIzLjY4LDI0LjExaDguMzF2LTguOTRoLTcuODZjLTguMjQuMTEtMTUtNi41LTE1LjEtMTQuNzRzNi41LTE1LDE0Ljc0LTE1LjFoLjM0YzguMjIsMCwxNC45NSw2LjY5LDE0Ljk1LDE0LjkzaDB2NDQuNjNoMGw4LjkyLjA2WiIvPgogICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNNTQuOCw0Ny45aDguOTJ2LTIzLjg4YzAtOC4yNCw2LjY5LTE0LjkzLDE0LjkzLTE0LjkzLDIuNzIsMCw1LjI1LjcyLDcuNDYsMmw0LjQ4LTcuNzVjLTMuNS0yLjAyLTcuNTgtMy4xOS0xMS45Mi0zLjE5LTEzLjE5LDAtMjMuODgsMTAuNjgtMjMuODgsMjMuODh2MjMuODhaIi8+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xOTguMDEuNzRjLjY4LjM4LDEuMjEuOTEsMS41OSwxLjU5LjM4LjY4LjU3LDEuNDIuNTcsMi4yNXMtLjE5LDEuNTctLjU5LDIuMjdjLS40LjY4LS45MywxLjIzLTEuNjEsMS42MS0uNjguNC0xLjQ0LjU5LTIuMjUuNTlzLTEuNTctLjE5LTIuMjUtLjU5Yy0uNjgtLjQtMS4yMS0uOTMtMS41OS0xLjYxLS4zOC0uNjgtLjU5LTEuNDItLjU5LTIuMjVzLjE5LTEuNTcuNTktMi4yNWMuMzgtLjY4LjkzLTEuMjEsMS42MS0xLjYxczEuNDQtLjU5LDIuMjctLjU5Yy44MywwLDEuNTcuMTksMi4yNS41OVpNMTk3LjU3LDcuNzVjLjU1LS4zMi45OC0uNzYsMS4zLTEuMzIuMzItLjU1LjQ3LTEuMTcuNDctMS44NXMtLjE1LTEuMy0uNDctMS44NS0uNzQtLjk4LTEuMjctMS4zYy0uNTUtLjMyLTEuMTctLjQ3LTEuODUtLjQ3cy0xLjMuMTctMS44NS40OWMtLjU1LjMyLS45OC43Ni0xLjMsMS4zMnMtLjQ3LDEuMTctLjQ3LDEuODUuMTUsMS4zLjQ3LDEuODVjLjMyLjU1Ljc0LDEsMS4yNywxLjMyLjU1LjMyLDEuMTUuNDksMS44My40OS43LS4wNCwxLjMyLS4yMSwxLjg3LS41M1pNMTk3Ljg0LDQuODJjLS4xNS4yNS0uMzguNDUtLjY4LjU5bDEuMDYsMS42NGgtMS4zMmwtLjkxLTEuNDJoLS44N3YxLjQyaC0xLjMyVjIuMTdoMi4xMmMuNjYsMCwxLjE5LjE1LDEuNTcuNDcuMzguMzIuNTcuNzQuNTcsMS4yNywwLC4zNC0uMDguNjYtLjIzLjkxWk0xOTUuODUsNC42NWMuMywwLC41My0uMDYuNjgtLjE5LjE3LS4xMy4yNS0uMzIuMjUtLjU1cy0uMDgtLjQyLS4yNS0uNTctLjQtLjE5LS42OC0uMTloLS43NHYxLjUzaC43NHYtLjAyWiIvPgogIDwvZz4KPC9zdmc+)\
Fonte: https://groq.com/

**Groq** (https://groq.com) é uma empresa americana de inteligência artificial fundada em 2016 por ex-engenheiros do Google. Seu principal diferencial e inovação reside no desenvolvimento de um circuito integrado específico para aplicações de IA que eles chamam de **LPU** (**Language Processing Unit**), e hardware relacionado.

A missão da Groq é acelerar o desempenho da inferência de cargas de trabalho de IA, ou seja, o processo de usar um modelo de IA já treinado para gerar previsões ou respostas. Eles se destacam por oferecer velocidade de processamento e eficiência incomparáveis, superando as GPUs (Graphics Processing Units) nesse aspecto, que foram originalmente projetadas para processamento gráfico e adaptadas para IA.

**Pontos Chave sobre o Groq:**

1. **LPU (Language Processing Unit)**: É o chip especializado da Groq. Diferente das GPUs, que são mais versáteis, as LPUs foram projetadas especificamente para a inferência de modelos de IA, especialmente Large Language Models (LLMs - Grandes Modelos de Linguagem). Essa especialização permite que as LPUs atinjam latências ultrabaixas e alto throughput (taxa de geração de tokens por segundo).
2. **Velocidade Instantânea**: A Groq tem ganhado destaque no mercado por sua capacidade de gerar respostas de LLMs quase instantaneamente. Eles frequentemente demonstram que seus sistemas podem gerar centenas ou até milhares de tokens por segundo, um desempenho significativamente mais rápido do que muitas outras soluções disponíveis.
3. **Eficiência Energética e Custo**: Além da velocidade, a arquitetura da LPU também é otimizada para maior eficiência energética e menor custo por inferência em comparação com as GPUs tradicionais.

\
Em resumo, a **Groq** se posiciona como uma alternativa poderosa à **NVIDIA** no espaço de hardware de IA, focando especificamente em oferecer a inferência de LLMs mais rápida e eficiente do mercado por meio de sua inovadora tecnologia LPU.

Até a geração desse material o Groq oferece API Key gratuitas para desenvolvedores testarem diversos tipos de modelos. Para conseguir uma API, você deve se registrar no Groq e criar uma Free API Key.

\

![groq](https://middleware.datah.ai/groq.gif)

# Anexo II - Secrets

Agora vamos configurar nossa API nos **Secrets** Google Colaboratory.

\
![secrets](https://middleware.datah.ai/secrets.gif)

#Anexo III - Referências

## LLM
https://groq.com/  
https://console.groq.com/docs/models  

## Frameworks
https://www.langchain.com/  
https://python.langchain.com/docs/integrations/vectorstores/  
https://python.langchain.com/docs/integrations/retrievers/  
https://python.langchain.com/docs/integrations/tools/  
https://python.langchain.com/docs/integrations/providers/huggingface/  

## Outros Frameworks
https://www.llamaindex.ai/  
https://www.crewai.com/  
https://google.github.io/adk-docs/get-started/  

## RAG

### Ferramenta no-code
https://ragflow.io/

### Arquiteturas de RAG (segundo link tem vários papers)
https://medium.com/@rupeshit/mastering-the-25-types-of-rag-architectures-when-and-how-to-use-each-one-2ca0e4b944d7  
https://www.marktechpost.com/2024/11/25/retrieval-augmented-generation-rag-deep-dive-into-25-different-types-of-rag/  
https://bhavishyapandit9.substack.com/p/25-types-of-rag-final-chapter  

## Context Engineering (Repositório muito legal)
https://github.com/davidkimai/Context-Engineering

## Outros assuntos que podem agregar:
A2A: https://a2aprotocol.ai/  
ANP: https://www.agent-network-protocol.com/  
ACP: https://www.ibm.com/think/topics/agent-communication-protocol  


